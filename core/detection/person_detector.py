"""
core/detection/person_detector.py
STEP 2 — Person Detector using YOLOv8

Takes a raw BGR frame from StreamReader and returns a list of
DetectedPerson objects — each with bounding box, confidence score,
and a cropped image of the person (needed by Re-ID in Step 4).

Design decisions:
  - Only detects class 0 (person) — ignores cars, bags, etc.
  - Returns crops alongside boxes so Re-ID doesn't need to re-crop
  - Draws annotations directly on a copy of the frame (non-destructive)
  - Accepts single frame OR list of frames (for multi-camera batch)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from loguru import logger

from config.settings import (
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    YOLO_DEVICE,
)

# Lazy import — ultralytics is only loaded when detector is first created
_ultralytics_available = False
try:
    from ultralytics import YOLO
    _ultralytics_available = True
except ImportError:
    logger.error("ultralytics not installed. Run: pip install ultralytics")


# ── Data container for one detected person ────────────────────────────────────

@dataclass
class DetectedPerson:
    """
    One person detected in a frame.

    Fields:
        bbox        : (x1, y1, x2, y2) in pixels — top-left and bottom-right
        confidence  : YOLOv8 detection confidence 0.0 → 1.0
        crop        : BGR image of just the person (used by Re-ID embedder)
        camera_id   : which camera this came from (set by caller)
        frame_num   : frame number from StreamReader
    """
    bbox        : Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence  : float
    crop        : np.ndarray                  # BGR person crop
    camera_id   : str  = "cam1"
    frame_num   : int  = 0

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

    @property
    def area(self) -> int: return self.width * self.height


# ── Person Detector ───────────────────────────────────────────────────────────

class PersonDetector:
    """
    Wraps YOLOv8 to detect only people in video frames.

    Usage:
        detector = PersonDetector()
        persons  = detector.detect(frame, camera_id="cam1", frame_num=42)
        annotated = detector.annotate(frame, persons)
    """

    PERSON_CLASS_ID = 0   # COCO class 0 = person

    def __init__(
        self,
        model_path : str   = YOLO_MODEL,
        confidence : float = YOLO_CONFIDENCE,
        device     : str   = YOLO_DEVICE,
        min_height : int   = 40,    # ignore tiny detections (likely far away / noise)
        min_width  : int   = 20,
    ):
        if not _ultralytics_available:
            raise RuntimeError("ultralytics package not installed. Run: pip install ultralytics")

        self.confidence = confidence
        self.device     = device
        self.min_height = min_height
        self.min_width  = min_width

        logger.info(f"Loading YOLOv8 model: {model_path} | device={device} | conf={confidence}")
        self._model = YOLO(model_path)
        self._model.to(device)
        logger.success(f"YOLOv8 model loaded: {model_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        frame     : np.ndarray,
        camera_id : str = "cam1",
        frame_num : int = 0,
    ) -> List[DetectedPerson]:
        """
        Run YOLOv8 on a single frame.
        Returns list of DetectedPerson (may be empty if no people found).

        Args:
            frame     : BGR image from StreamReader
            camera_id : which camera this frame came from
            frame_num : frame number for tracking/logging

        Returns:
            List[DetectedPerson] sorted by confidence (highest first)
        """
        if frame is None or frame.size == 0:
            logger.warning(f"[{camera_id}] Empty frame received — skipping detection.")
            return []

        # Run inference — verbose=False suppresses per-frame console spam
        results = self._model(
            frame,
            conf    = self.confidence,
            classes = [self.PERSON_CLASS_ID],
            device  = self.device,
            verbose = False,
        )

        persons = []
        for result in results:
            for box in result.boxes:
                # Extract bbox as integers
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Filter out detections that are too small
                w = x2 - x1
                h = y2 - y1
                if w < self.min_width or h < self.min_height:
                    continue

                # Clamp to frame boundaries
                fh, fw = frame.shape[:2]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(fw, x2); y2 = min(fh, y2)

                # Crop person out of frame for Re-ID
                crop = frame[y1:y2, x1:x2].copy()

                persons.append(DetectedPerson(
                    bbox       = (x1, y1, x2, y2),
                    confidence = conf,
                    crop       = crop,
                    camera_id  = camera_id,
                    frame_num  = frame_num,
                ))

        # Sort by confidence descending
        persons.sort(key=lambda p: p.confidence, reverse=True)

        logger.debug(f"[{camera_id}] Frame #{frame_num} → {len(persons)} person(s) detected.")
        return persons

    def annotate(
        self,
        frame   : np.ndarray,
        persons : List[DetectedPerson],
        color   : Tuple[int,int,int] = (0, 255, 0),
        show_confidence : bool = True,
        show_center     : bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a COPY of the frame.
        Does NOT modify the original frame.

        Returns annotated BGR image.
        """
        annotated = frame.copy()

        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person.bbox

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label: "Person  0.87"
            label = f"Person"
            if show_confidence:
                label += f"  {person.confidence:.2f}"

            # Background for label text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

            # Center dot
            if show_center:
                cv2.circle(annotated, person.center, 4, (0, 0, 255), -1)

        # HUD — person count top-left
        hud = f"Persons: {len(persons)}"
        cv2.putText(annotated, hud, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def detect_and_annotate(
        self,
        frame     : np.ndarray,
        camera_id : str = "cam1",
        frame_num : int = 0,
    ) -> Tuple[List[DetectedPerson], np.ndarray]:
        """
        Convenience: detect + annotate in one call.
        Returns (persons, annotated_frame).
        """
        persons  = self.detect(frame, camera_id=camera_id, frame_num=frame_num)
        annotated = self.annotate(frame, persons)
        return persons, annotated


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from core.video.stream_reader import StreamReader

    source = sys.argv[1] if len(sys.argv) > 1 else "data/cam1.mp4"
    print(f"\nStep 2 — Person Detector test")
    print(f"Source : {source}")
    print(f"Model  : {YOLO_MODEL}  (auto-downloads on first run ~6MB)")
    print(f"Device : {YOLO_DEVICE}")
    print("Press Q to quit.\n")

    detector = PersonDetector()
    reader   = StreamReader(source=source, frame_skip=1)

    try:
        for frame_num, frame in reader.frames():
            persons, annotated = detector.detect_and_annotate(
                frame,
                camera_id="cam1",
                frame_num=frame_num,
            )

            # Print summary every 30 frames
            if frame_num % 30 == 0:
                print(f"Frame #{frame_num:5d} | {len(persons)} person(s) detected")
                for i, p in enumerate(persons):
                    print(f"  Person {i+1}: bbox={p.bbox}  conf={p.confidence:.2f}  "
                          f"crop={p.crop.shape}")

            cv2.imshow("URG-IS | Step 2 — Person Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("Done.")