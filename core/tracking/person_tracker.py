"""
core/tracking/person_tracker.py
STEP 3 — Person Tracker using BoT-SORT

Takes detected persons from PersonDetector and assigns each a
consistent track ID across frames. The same physical person
keeps the same ID even when they move around the frame.

How it works:
  - BoT-SORT (Built-in Ultralytics tracker) combines:
      * Motion prediction (Kalman filter) — predicts where person will be next frame
      * Appearance matching — matches visual features between frames
      * IoU matching — matches bounding box overlap
  - Each person gets a track_id (integer) that persists across frames
  - If person disappears behind obstacle and reappears within
    BOTSORT_BUFFER_FRAMES, same track_id is reused

Design decisions:
  - We run YOLOv8 track() instead of predict() — tracking is built into ultralytics
  - Returns TrackedPerson objects that extend DetectedPerson with track_id
  - Buffer set to 90 frames (3 seconds at 30fps) per report §12
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from loguru import logger

from config.settings import (
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
    BOTSORT_BUFFER_FRAMES,
)

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class TrackedPerson:
    """
    A person detected AND tracked across frames.
    Extends detection with a stable track_id.

    track_id  : integer assigned by BoT-SORT, consistent across frames
    is_new    : True if this track_id appeared for the first time this frame
    """
    track_id   : int
    bbox       : Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence : float
    crop       : np.ndarray
    camera_id  : str = "cam1"
    frame_num  : int = 0
    is_new     : bool = False                # first time we see this track_id

    @property
    def x1(self): return self.bbox[0]
    @property
    def y1(self): return self.bbox[1]
    @property
    def x2(self): return self.bbox[2]
    @property
    def y2(self): return self.bbox[3]

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self)  -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1


# ── Person Tracker ────────────────────────────────────────────────────────────

class PersonTracker:
    """
    Wraps YOLOv8 BoT-SORT tracking.
    Maintains track history and detects new vs existing tracks.

    Usage:
        tracker = PersonTracker()
        tracked = tracker.track(frame, camera_id="cam1", frame_num=42)
        annotated = tracker.annotate(frame, tracked)
    """

    PERSON_CLASS_ID = 0
    # Distinct colours per track_id for visualisation
    TRACK_COLORS = [
        (0,255,0),(255,128,0),(0,128,255),(255,0,255),
        (0,255,255),(255,255,0),(128,0,255),(255,0,128),
        (0,200,100),(100,200,0),(200,100,0),(0,100,200),
    ]

    def __init__(
        self,
        model_path    : str   = YOLO_MODEL,
        confidence    : float = YOLO_CONFIDENCE,
        device        : str   = YOLO_DEVICE,
        buffer_frames : int   = BOTSORT_BUFFER_FRAMES,
        min_height    : int   = 40,
        min_width     : int   = 20,
    ):
        self.confidence    = confidence
        self.device        = device
        self.buffer_frames = buffer_frames
        self.min_height    = min_height
        self.min_width     = min_width

        # Track history: track_id → list of center points (for trail drawing)
        self._track_history  : Dict[int, List[Tuple[int,int]]] = {}
        # Known track IDs so we can flag new ones
        self._known_track_ids: set = set()

        logger.info(f"Loading YOLOv8 tracker: {model_path} | device={device} | "
                    f"buffer={buffer_frames} frames")
        self._model = YOLO(model_path)
        self._model.to(device)
        logger.success("PersonTracker ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def track(
        self,
        frame     : np.ndarray,
        camera_id : str = "cam1",
        frame_num : int = 0,
    ) -> List[TrackedPerson]:
        """
        Run BoT-SORT tracking on a single frame.
        Returns list of TrackedPerson with stable track_ids.

        Args:
            frame     : BGR image from StreamReader
            camera_id : which camera (used for logging)
            frame_num : frame number

        Returns:
            List[TrackedPerson] — empty if no people tracked
        """
        if frame is None or frame.size == 0:
            return []

        # Run YOLOv8 with BoT-SORT tracker
        # persist=True keeps tracker state between calls (essential!)
        results = self._model.track(
            frame,
            conf      = self.confidence,
            classes   = [self.PERSON_CLASS_ID],
            tracker   = "botsort.yaml",
            persist   = True,
            device    = self.device,
            verbose   = False,
        )

        tracked = []

        for result in results:
            if result.boxes.id is None:
                # No tracks this frame (tracker not yet initialised or empty scene)
                continue

            boxes    = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confs    = result.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)

                # Filter tiny detections
                if (x2 - x1) < self.min_width or (y2 - y1) < self.min_height:
                    continue

                # Clamp to frame
                fh, fw = frame.shape[:2]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(fw, x2); y2 = min(fh, y2)

                crop   = frame[y1:y2, x1:x2].copy()
                is_new = track_id not in self._known_track_ids

                person = TrackedPerson(
                    track_id   = int(track_id),
                    bbox       = (x1, y1, x2, y2),
                    confidence = float(conf),
                    crop       = crop,
                    camera_id  = camera_id,
                    frame_num  = frame_num,
                    is_new     = is_new,
                )
                tracked.append(person)

                # Update history
                self._known_track_ids.add(int(track_id))
                cx, cy = person.center
                if track_id not in self._track_history:
                    self._track_history[track_id] = []
                self._track_history[track_id].append((cx, cy))

                # Keep trail to last 60 points
                if len(self._track_history[track_id]) > 60:
                    self._track_history[track_id].pop(0)

                if is_new:
                    logger.info(f"[{camera_id}] Frame #{frame_num} → "
                                f"NEW track ID {track_id} appeared.")

        logger.debug(f"[{camera_id}] Frame #{frame_num} → "
                     f"{len(tracked)} tracked person(s).")
        return tracked

    def annotate(
        self,
        frame   : np.ndarray,
        tracked : List[TrackedPerson],
        show_trail      : bool = True,
        show_confidence : bool = True,
    ) -> np.ndarray:
        """
        Draw tracking annotations on a COPY of the frame.

        Each track gets a unique colour.
        A motion trail shows where the person has been.
        """
        annotated = frame.copy()

        for person in tracked:
            color = self._track_color(person.track_id)
            x1, y1, x2, y2 = person.bbox

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label: "ID:3  0.88"
            label = f"ID:{person.track_id}"
            if show_confidence:
                label += f"  {person.confidence:.2f}"
            if person.is_new:
                label += "  NEW"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

            # Center dot
            cv2.circle(annotated, person.center, 4, color, -1)

            # Motion trail
            if show_trail and person.track_id in self._track_history:
                pts = self._track_history[person.track_id]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)   # fade older points
                    c = tuple(int(v * alpha) for v in color)
                    cv2.line(annotated, pts[i-1], pts[i], c, 2)

        # HUD
        cv2.putText(annotated,
                    f"Tracking: {len(tracked)} person(s)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def track_and_annotate(
        self,
        frame     : np.ndarray,
        camera_id : str = "cam1",
        frame_num : int = 0,
    ) -> Tuple[List[TrackedPerson], np.ndarray]:
        """Convenience: track + annotate in one call."""
        tracked   = self.track(frame, camera_id=camera_id, frame_num=frame_num)
        annotated = self.annotate(frame, tracked)
        return tracked, annotated

    def get_track_history(self, track_id: int) -> List[Tuple[int,int]]:
        """Returns list of (cx, cy) center points for a given track_id."""
        return self._track_history.get(track_id, [])

    def reset(self):
        """Clear all track history (call when switching video source)."""
        self._track_history.clear()
        self._known_track_ids.clear()
        logger.info("PersonTracker reset.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _track_color(self, track_id: int) -> Tuple[int,int,int]:
        """Returns a consistent colour for a given track_id."""
        return self.TRACK_COLORS[track_id % len(self.TRACK_COLORS)]


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from core.video.stream_reader import StreamReader

    source = sys.argv[1] if len(sys.argv) > 1 else "data/cam1.mp4"
    print(f"\nStep 3 — Person Tracker test")
    print(f"Source : {source}")
    print(f"Each person gets a unique colour + motion trail.")
    print("Press Q to quit.\n")

    tracker = PersonTracker()
    reader  = StreamReader(source=source, frame_skip=1)

    try:
        for frame_num, frame in reader.frames():
            tracked, annotated = tracker.track_and_annotate(
                frame,
                camera_id="cam1",
                frame_num=frame_num,
            )

            if frame_num % 30 == 0:
                print(f"Frame #{frame_num:5d} | tracking {len(tracked)} person(s)")
                for p in tracked:
                    status = "NEW" if p.is_new else "   "
                    print(f"  [{status}] ID:{p.track_id:3d}  "
                          f"conf={p.confidence:.2f}  center={p.center}")

            cv2.imshow("URG-IS | Step 3 — Person Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("Done.")