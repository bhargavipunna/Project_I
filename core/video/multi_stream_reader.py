"""
core/video/multi_stream_reader.py
STEP 1 (Complete) — Multi-Camera Stream Reader

Reads all 7 WILDTRACK cameras in sync using threads.
Each camera runs in its own thread, frames are collected
into a shared queue so the pipeline always gets a full
set of frames (one per camera) at the same time.

Usage:
    reader = MultiCameraStreamReader("data/Wildtrack/")
    for frame_set in reader.frame_sets():
        # frame_set = {
        #   "cam1": (frame_num, frame_bgr),
        #   "cam2": (frame_num, frame_bgr),
        #   ...
        # }
        process(frame_set)
"""

import cv2
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
from loguru import logger

from config.settings import (
    DETECTION_WIDTH,
    DETECTION_HEIGHT,
    FRAME_SKIP,
)
from core.video.stream_reader import StreamReader


# ── Data container for one camera's frame ────────────────────────────────────

class CameraFrame:
    def __init__(self, camera_id: str, frame_num: int, frame):
        self.camera_id = camera_id
        self.frame_num = frame_num
        self.frame     = frame


# ── Multi-Camera Reader ───────────────────────────────────────────────────────

class MultiCameraStreamReader:
    """
    Reads multiple camera sources simultaneously using one thread per camera.
    Designed specifically for WILDTRACK (7 cams) but works for any number.

    Frame sync strategy:
      - Each camera thread pushes frames into a shared queue
      - Main thread collects one frame per camera into a "frame set"
      - Frame sets are yielded to the pipeline in order
      - If a camera is slow/missing, it is skipped gracefully
    """

    WILDTRACK_CAMS = [
        "cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7"
    ]

    def __init__(
        self,
        source_dir: str,
        camera_ids: Optional[List[str]] = None,
        frame_width:  int = DETECTION_WIDTH,
        frame_height: int = DETECTION_HEIGHT,
        frame_skip:   int = FRAME_SKIP,
        max_queue_size: int = 30,        # frames buffered per camera
        sync_timeout:   float = 2.0,     # seconds to wait for slow camera
    ):
        self.source_dir   = Path(source_dir)
        self.camera_ids   = camera_ids or self.WILDTRACK_CAMS
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.frame_skip   = frame_skip
        self.sync_timeout = sync_timeout

        # One queue per camera
        self._queues: Dict[str, queue.Queue] = {
            cam_id: queue.Queue(maxsize=max_queue_size)
            for cam_id in self.camera_ids
        }

        self._stop_event  = threading.Event()
        self._threads: List[threading.Thread] = []
        self._readers: List[StreamReader] = []

        logger.info(
            f"MultiCameraStreamReader | cameras={self.camera_ids} | "
            f"source_dir={self.source_dir}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def frame_sets(self) -> Generator[Dict[str, CameraFrame], None, None]:
        """
        Generator that yields a dict of {camera_id: CameraFrame} for each tick.
        All cameras must contribute a frame for the set to be yielded.

        Usage:
            for frame_set in reader.frame_sets():
                for cam_id, cam_frame in frame_set.items():
                    cv2.imshow(cam_id, cam_frame.frame)
        """
        self._start_camera_threads()

        try:
            while not self._stop_event.is_set():
                frame_set = {}
                all_done  = True

                for cam_id in self.camera_ids:
                    try:
                        cam_frame = self._queues[cam_id].get(timeout=self.sync_timeout)

                        if cam_frame is None:
                            # None is the "stream finished" sentinel
                            logger.info(f"{cam_id} stream finished.")
                            continue

                        frame_set[cam_id] = cam_frame
                        all_done = False

                    except queue.Empty:
                        logger.warning(f"{cam_id} frame timeout — skipping this camera for tick")

                if all_done:
                    logger.info("All camera streams finished.")
                    break

                if frame_set:
                    yield frame_set

        finally:
            self.stop()

    def stop(self):
        """Stop all camera threads."""
        self._stop_event.set()
        for reader in self._readers:
            reader.stop()
        for thread in self._threads:
            thread.join(timeout=3.0)
        logger.info("MultiCameraStreamReader stopped.")

    def get_active_cameras(self) -> List[str]:
        """Returns list of camera IDs that have valid source files."""
        active = []
        for cam_id in self.camera_ids:
            path = self._resolve_source(cam_id)
            if path and Path(path).exists():
                active.append(cam_id)
        return active

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_source(self, camera_id: str) -> Optional[str]:
        """
        Resolves camera_id to a file path.
        Supports:
          - "cam1" → looks for cam1.mp4 / cam1.avi in source_dir
          - Full path string
          - RTSP URL (passed through as-is)
        """
        # Already a full path or RTSP
        if camera_id.startswith("rtsp://") or Path(camera_id).exists():
            return camera_id

        # Search source_dir for matching file
        for ext in [".mp4", ".avi", ".mkv", ".mov"]:
            candidate = self.source_dir / f"{camera_id}{ext}"
            if candidate.exists():
                return str(candidate)

        logger.warning(f"No video file found for camera '{camera_id}' in {self.source_dir}")
        return None

    def _camera_thread_fn(self, camera_id: str, source: str):
        """
        Each camera runs in its own thread.
        Reads frames from StreamReader and pushes them into the camera's queue.
        Pushes None sentinel when stream ends.
        """
        reader = StreamReader(
            source=source,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            frame_skip=self.frame_skip,
        )
        self._readers.append(reader)

        logger.info(f"Camera thread started: {camera_id} | source={source}")

        try:
            for frame_num, frame in reader.frames():
                if self._stop_event.is_set():
                    break
                cam_frame = CameraFrame(camera_id, frame_num, frame)
                try:
                    self._queues[camera_id].put(cam_frame, timeout=1.0)
                except queue.Full:
                    # Drop oldest frame if queue is full (pipeline too slow)
                    try:
                        self._queues[camera_id].get_nowait()
                        self._queues[camera_id].put_nowait(cam_frame)
                    except queue.Empty:
                        pass
        finally:
            # Push sentinel to signal stream end
            try:
                self._queues[camera_id].put_nowait(None)
            except queue.Full:
                pass
            logger.info(f"Camera thread finished: {camera_id}")

    def _start_camera_threads(self):
        """Starts one reader thread per camera."""
        self._threads = []
        self._readers = []

        for cam_id in self.camera_ids:
            source = self._resolve_source(cam_id)
            if source is None:
                logger.warning(f"Skipping {cam_id} — no source found.")
                # Push sentinel immediately so frame_sets() doesn't hang
                self._queues[cam_id].put_nowait(None)
                continue

            t = threading.Thread(
                target=self._camera_thread_fn,
                args=(cam_id, source),
                daemon=True,
                name=f"camera-{cam_id}",
            )
            self._threads.append(t)
            t.start()

        logger.success(f"Started {len(self._threads)} camera threads.")


# ── Smoke-test: preview all cameras in a grid ─────────────────────────────────
if __name__ == "__main__":
    import sys
    import numpy as np

    source_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Wildtrack"

    print(f"\nTesting MultiCameraStreamReader with: {source_dir}")
    print("Shows all cameras in a grid. Press Q to quit.\n")

    reader = MultiCameraStreamReader(source_dir=source_dir, frame_skip=1)
    active = reader.get_active_cameras()
    print(f"Active cameras found: {active}\n")

    if not active:
        print(f"ERROR: No video files found in '{source_dir}'")
        print("Make sure cam1.mp4 ... cam7.mp4 are in that folder.")
        sys.exit(1)

    # Build a grid layout: 4 cameras top row, 3 bottom row
    GRID_W, GRID_H = 320, 240   # size per camera tile

    try:
        for frame_set in reader.frame_sets():
            tiles = []
            for cam_id in active:
                if cam_id not in frame_set:
                    tile = np.zeros((GRID_H, GRID_W, 3), dtype="uint8")
                else:
                    tile = cv2.resize(frame_set[cam_id].frame, (GRID_W, GRID_H))
                    fn   = frame_set[cam_id].frame_num
                    # Label each tile
                    cv2.putText(tile, cam_id,  (8, 22),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(tile, f"f#{fn}", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                tiles.append(tile)

            # Pad to 8 tiles for even 4+4 grid
            while len(tiles) % 4 != 0:
                tiles.append(np.zeros((GRID_H, GRID_W, 3), dtype="uint8"))

            row1 = np.hstack(tiles[:4])
            row2 = np.hstack(tiles[4:8])
            grid = np.vstack([row1, row2])

            cv2.imshow("URG-IS | All Cameras", grid)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("Done.")