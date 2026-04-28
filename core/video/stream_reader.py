"""
core/video/stream_reader.py
STEP 1 — Video Stream Reader

Reads frames from:
  - RTSP stream (live CCTV)
  - Local video file (for development/testing)
  - Webcam (index 0, 1, ...)

Design decisions:
  - Generator pattern: callers use `for frame in reader.frames()` — clean, memory-efficient
  - Frame skipping: only every Nth frame is yielded (CPU optimisation from report §12)
  - Auto-reconnect: if RTSP drops, reader waits and retries instead of crashing
  - Resize on capture: all frames normalised to configured resolution before any processing
"""

import cv2
import time
import threading
import os
from typing import Generator, Optional, Tuple
from loguru import logger

from config.settings import (
    DETECTION_WIDTH,
    DETECTION_HEIGHT,
    FRAME_SKIP,
)


class StreamReader:
    """
    Wraps OpenCV VideoCapture with:
      - Configurable source (RTSP / file / webcam)
      - Frame skipping for CPU efficiency
      - Auto-reconnect on RTSP stream loss
      - Thread-safe stop signal
    """

    def __init__(
        self,
        source: Optional[str] = None,
        frame_width: int = DETECTION_WIDTH,
        frame_height: int = DETECTION_HEIGHT,
        frame_skip: int = FRAME_SKIP,
        reconnect_delay: int = int(os.getenv("RTSP_RECONNECT_DELAY", "3")),
    ):
        # Allow integer string "0" → int for webcam
        if source is None:
            source = os.getenv("VIDEO_SOURCE", "0")
        try:
            self.source = int(source)
        except (ValueError, TypeError):
            self.source = source

        self.frame_width    = frame_width
        self.frame_height   = frame_height
        self.frame_skip     = frame_skip
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._stop_event = threading.Event()

        logger.info(f"StreamReader initialised | source={self.source} | "
                    f"resolution={frame_width}x{frame_height} | frame_skip={frame_skip}")

    # ── Public API ────────────────────────────────────────────────────────────

    def frames(self) -> Generator[Tuple[int, any], None, None]:
        """
        Generator that yields (frame_number, frame_bgr).
        Handles reconnection automatically for RTSP sources.

        Usage:
            reader = StreamReader("rtsp://192.168.1.10/stream1")
            for frame_num, frame in reader.frames():
                process(frame)
        """
        frame_count = 0

        while not self._stop_event.is_set():
            self._cap = self._open_capture()

            if self._cap is None:
                logger.warning(f"Could not open source. Retrying in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                continue

            logger.success(f"Stream opened: {self.source}")

            while not self._stop_event.is_set():
                ret, frame = self._cap.read()

                if not ret:
                    logger.warning("Frame read failed — stream ended or connection lost.")
                    break  # triggers outer reconnect loop

                frame_count += 1

                # Skip frames for CPU efficiency (report §12 — "skip every 2 frames")
                if frame_count % (self.frame_skip + 1) != 0:
                    continue

                # Normalise resolution
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                yield frame_count, frame

            # Clean up before potential reconnect
            if self._cap:
                self._cap.release()
                self._cap = None

            if not self._stop_event.is_set():
                logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)

    def stop(self):
        """Signal the frame generator to stop."""
        self._stop_event.set()
        if self._cap:
            self._cap.release()
        logger.info("StreamReader stopped.")

    def get_fps(self) -> float:
        """Returns source FPS if available (useful for duration calculations)."""
        if self._cap and self._cap.isOpened():
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0

    def get_frame_size(self) -> Tuple[int, int]:
        return self.frame_width, self.frame_height

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """
        Opens VideoCapture with RTSP optimisation flags when source is a URL.
        Returns None on failure so caller can retry.
        """
        try:
            if isinstance(self.source, str) and self.source.startswith("rtsp://"):
                # Use FFmpeg backend for RTSP — more reliable than GStreamer
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                # Reduce internal buffer to minimise latency (live CCTV should be real-time)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                logger.error(f"VideoCapture.open() failed for source: {self.source}")
                return None

            return cap

        except Exception as e:
            logger.error(f"Exception opening capture: {e}")
            return None


# ── Quick smoke-test (run this file directly) ─────────────────────────────────
if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    print(f"\nTesting StreamReader with source: '{source}'")
    print("Press Q in the preview window to quit.\n")

    reader = StreamReader(source=source, frame_skip=1)

    try:
        for frame_num, frame in reader.frames():
            print(f"\rFrame #{frame_num}", end="", flush=True)

            # Draw frame number on preview
            cv2.putText(
                frame,
                f"Frame: {frame_num}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("URG-IS | Stream Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("Done.")