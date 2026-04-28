"""
core/pipeline.py
Main Pipeline Orchestrator — wires Steps 1-6 and feeds Step 7 (API)

Runs all steps in a loop for each camera frame:
  1. StreamReader      → raw frames
  2. PersonTracker     → tracked persons with stable IDs
  3. PersonEmbedder    → appearance embeddings
  4. IdentityManager   → PERSON_IDs (cross-camera stable)
  5. InteractionDetector → proximity events
  6. ConfidenceEngine  → incident classification + graph update
  7. API notify        → push to WebSocket clients

Supports:
  - Single camera
  - All 7 WILDTRACK cameras simultaneously
"""

import asyncio
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict
from loguru import logger

import cv2

from core.video.stream_reader import StreamReader
from core.video.multi_stream_reader import MultiCameraStreamReader
from core.tracking.person_tracker import PersonTracker
from core.reid.fusion_embedder import FusionEmbedder
from core.reid.identity_manager import IdentityManager
from core.interaction.interaction_detector import InteractionDetector, annotate_interactions
from core.graph.graph_db import GraphDB
from core.graph.confidence_engine import ConfidenceEngine
from core.spatial.floor_mapper import UnifiedFloorMapper
from core.streaming.redis_streams import RedisIncidentStream

from config.settings import ENABLE_REDIS_STREAMS, ENABLE_UNIFIED_FLOOR_MAP, FRAME_SKIP


class Pipeline:
    """
    Full URG-IS pipeline.

    Usage — single camera:
        pipeline = Pipeline(source="data/cam1.mp4")
        pipeline.run()

    Usage — all 7 cameras:
        pipeline = Pipeline(source="data/", multi_camera=True)
        pipeline.run()

    Usage — with API (background thread):
        pipeline = Pipeline(source="data/cam1.mp4", api_notify_fn=notify_graph_updated)
        thread = threading.Thread(target=pipeline.run)
        thread.start()
    """

    def __init__(
        self,
        source          : str = "data/cam1.mp4",
        multi_camera    : bool = False,
        show_preview    : bool = True,
        api_notify_fn   = None,   # async fn to call after graph updates
        frame_skip      : int  = FRAME_SKIP,
    ):
        self.source        = source
        self.multi_camera  = multi_camera
        self.show_preview  = show_preview
        self._notify       = api_notify_fn
        self._stop_flag    = threading.Event()

        logger.info(f"Initialising pipeline | source={source} | multi={multi_camera}")

        # Initialise all components
        self.tracker  = PersonTracker()
        self.embedder = FusionEmbedder()
        self.manager  = IdentityManager()
        self.db       = GraphDB()
        self.engine   = ConfidenceEngine(self.db, decay_interval_m=10, auto_snapshot=True)
        self.floor_mapper = UnifiedFloorMapper() if ENABLE_UNIFIED_FLOOR_MAP else None
        self.redis_stream = RedisIncidentStream() if ENABLE_REDIS_STREAMS else None

        # One detector per camera
        self._detectors: Dict[str, InteractionDetector] = {}

        self._frame_count    = 0
        self._incident_count = 0
        self._loop           = None   # set by run.py before starting thread

        logger.success("Pipeline ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self):
        """Start pipeline. Blocks until stopped."""
        self.engine.start()
        try:
            if self.multi_camera:
                self._run_multi_camera()
            else:
                self._run_single_camera()
        finally:
            self._shutdown()

    def stop(self):
        """Signal pipeline to stop."""
        self._stop_flag.set()

    # ── Single camera loop ────────────────────────────────────────────────────

    def _run_single_camera(self):
        logger.info(f"Single camera pipeline: {self.source}")
        reader = StreamReader(source=self.source)
        try:
            for frame_num, frame in reader.frames():
                if self._stop_flag.is_set():
                    break
                self._process_frame(frame, camera_id="cam1", frame_num=frame_num)
        finally:
            reader.stop()
        logger.info("Video ended — pipeline stopping.")
        self.stop()

    # ── Multi camera loop ─────────────────────────────────────────────────────

    def _run_multi_camera(self):
        reader = MultiCameraStreamReader(source_dir=self.source)
        logger.info(f"Multi-camera pipeline: {reader.get_active_cameras()}")

        try:
            for frame_set in reader.frame_sets():
                if self._stop_flag.is_set():
                    break
                for cam_id, cam_frame in frame_set.items():
                    self._process_frame(
                        cam_frame.frame,
                        camera_id = cam_id,
                        frame_num = cam_frame.frame_num,
                    )
        finally:
            reader.stop()

    # ── Core frame processing ─────────────────────────────────────────────────

    def _process_frame(self, frame, camera_id: str, frame_num: int):
        """
        Process one frame through the full pipeline.
        Steps 3 → 4 → 5 → 6 → notify API.
        """
        self._frame_count += 1

        # Step 3 — Track persons
        tracked = self.tracker.track(frame, camera_id=camera_id, frame_num=frame_num)
        if not tracked:
            return

        # Step 4 — Re-identify persons
        persons_with_ids = []
        annotated        = frame.copy() if self.show_preview else None

        for person in tracked:
            vec    = self.embedder.embed(person.crop)
            result = self.manager.identify(
                embedding = vec,
                track_id  = person.track_id,
                camera_id = camera_id,
                frame_num = frame_num,
            )
            if result:
                persons_with_ids.append((result.person_id, person.center))

                if self.show_preview and annotated is not None:
                    x1, y1, x2, y2 = person.bbox
                    color = (0, 255, 0)
                    cv2.rectangle(annotated, (x1,y1),(x2,y2), color, 2)
                    cv2.putText(annotated,
                                f"P{result.person_id}",
                                (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if not persons_with_ids:
            return

        # Step 5 — Detect interactions
        detector = self._get_detector(camera_id)
        events   = detector.update(
            persons   = persons_with_ids,
            camera_id = camera_id,
            frame_num = frame_num,
        )

        # Step 6 — Process incidents + update graph
        people_ids     = [pid for pid, _ in persons_with_ids]
        persons_updated = set()

        for event in events:
            # Skip self-interaction (same person assigned two track IDs)
            if event.person_id_a == event.person_id_b:
                continue
            # Add location midpoint to event for location modifier
            pids   = [pid for pid, _ in persons_with_ids]
            centers = {pid: c for pid, c in persons_with_ids}
            ca     = centers.get(event.person_id_a, (0,0))
            cb     = centers.get(event.person_id_b, (0,0))
            event._location_px = ((ca[0]+cb[0])//2, (ca[1]+cb[1])//2)

            edge = self.engine.process_event(event, people_in_scene=people_ids)
            self._incident_count += 1
            persons_updated.add(event.person_id_a)
            persons_updated.add(event.person_id_b)

            if self.redis_stream and self.redis_stream.enabled:
                self.redis_stream.publish_incident(
                    {
                        "camera_id": event.camera_id,
                        "frame_num": event.frame_num,
                        "person_id_a": event.person_id_a,
                        "person_id_b": event.person_id_b,
                        "duration_s": event.duration_s,
                        "distance_m": event.distance_m,
                        "confidence": edge.confidence,
                        "relationship": edge.relationship,
                        "incident_type": edge.last_incident,
                    }
                )

        # Notify API (async, non-blocking)
        if events and self._notify and self._loop:
            floor_points = []
            if self.floor_mapper:
                for pid, center in persons_with_ids:
                    p = self.floor_mapper.make_point(pid, camera_id, center)
                    if p is not None:
                        floor_points.append(
                            {
                                "person_id": p.person_id,
                                "camera_id": p.camera_id,
                                "world_xy_m": [round(p.world_xy_m[0], 3), round(p.world_xy_m[1], 3)],
                                "map_xy_px": [int(p.map_xy_px[0]), int(p.map_xy_px[1])],
                            }
                        )
            asyncio.run_coroutine_threadsafe(
                self._notify(list(persons_updated)),
                self._loop,
            )

        # Preview window
        if self.show_preview and annotated is not None:
            if events:
                annotated = annotate_interactions(
                    annotated,
                    persons_with_ids,
                    detector.get_active_proximities(),
                    events,
                )
            stats = self.engine.get_stats()
            cv2.putText(annotated,
                        f"[{camera_id}] People:{len(persons_with_ids)} "
                        f"Identities:{self.manager.get_identity_count()} "
                        f"Relations:{stats['edges']}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
            cv2.imshow(f"URG-IS | {camera_id}", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop()

    def _get_detector(self, camera_id: str) -> InteractionDetector:
        """One InteractionDetector per camera."""
        if camera_id not in self._detectors:
            self._detectors[camera_id] = InteractionDetector()
        return self._detectors[camera_id]

    def _shutdown(self):
        self.engine.stop()
        self.manager.save()
        cv2.destroyAllWindows()
        logger.info(
            f"Pipeline stopped | "
            f"frames={self._frame_count} | "
            f"incidents={self._incident_count}"
        )


# ── Run standalone (no API) ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    source       = sys.argv[1] if len(sys.argv) > 1 else "data/cam1.mp4"
    multi_camera = "--multi" in sys.argv

    print(f"\nURG-IS Pipeline")
    print(f"Source       : {source}")
    print(f"Multi-camera : {multi_camera}")
    print("Press Q in preview window to stop.\n")

    pipeline = Pipeline(
        source       = source,
        multi_camera = multi_camera,
        show_preview = True,
    )
    pipeline.run()

    # Final per-person graph summary
    print("\n══ FINAL RELATIONSHIP GRAPHS ════════════════════════════════════")
    for pid, graph in pipeline.engine.get_all_person_graphs().items():
        if graph["total_connections"] == 0:
            continue
        print(f"\n▶ {pid} | {graph['total_connections']} connection(s) | "
              f"cameras: {graph['camera_ids']}")
        for conn in graph["connections"]:
            bar = "█" * int(conn["confidence"] * 20)
            print(f"  → {conn['person_id']:<15} "
                  f"[{conn['relationship']:<16}] "
                  f"{conn['confidence']:.3f} {bar}")
            print(f"     {conn['incident_counts']}  "
                  f"avg_dur={conn['avg_duration_s']}s")