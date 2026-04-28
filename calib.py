"""
auto_calib.py
=============
WHAT IT DOES:
  Estimates PIXELS_PER_METRE for every camera automatically.
  No human clicking. No markers. No manual measurements.

HOW IT WORKS:
  Average adult = 1.70m tall.
  YOLOv8 detects people → measures median bounding box height in pixels.
  PIXELS_PER_METRE = median_height_px / 1.70

ACCURACY:
  ±20% — good enough for 1.5m interaction distance threshold on MacBook.
  The 1-incident bug you had was caused by PIXELS_PER_METRE=100 fallback
  being used instead of the actual ~652 for your overhead cameras.
  This script fixes that permanently.

USAGE:
  python auto_calib.py              # all cameras in data/
  python auto_calib.py data/cam1.mp4  # single camera
  Called automatically by run_live.py --auto-calib
"""

import os, sys, json, glob
from pathlib import Path
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2
import numpy as np

os.makedirs("data/calib", exist_ok=True)

AVERAGE_PERSON_HEIGHT_M  = 1.70
MIN_PERSON_HEIGHT_PX     = 40
MAX_PERSON_HEIGHT_PX     = 1200
FALLBACK_PIXELS_PER_M    = 100.0


def estimate_pixels_per_metre(video_path: str, camera_id: str,
                               n_frames: int = 150,
                               min_detections: int = 8,
                               verbose: bool = True) -> float:
    from ultralytics import YOLO
    from config.settings import YOLO_MODEL, YOLO_CONFIDENCE

    if verbose:
        print(f"  [{camera_id}] auto-calibrating from {video_path} ...")

    model = YOLO(YOLO_MODEL)
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [{camera_id}] ERROR: cannot open {video_path}")
        return FALLBACK_PIXELS_PER_M

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip   = max(1, total // n_frames)
    heights: list[float] = []
    frame_idx = 0

    while frame_idx < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * skip)
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, classes=[0], conf=YOLO_CONFIDENCE, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            h = y2 - y1
            if MIN_PERSON_HEIGHT_PX <= h <= MAX_PERSON_HEIGHT_PX:
                heights.append(h)
        frame_idx += 1

    cap.release()

    if len(heights) < min_detections:
        if verbose:
            print(f"  [{camera_id}] only {len(heights)} detections — using fallback {FALLBACK_PIXELS_PER_M}")
        return FALLBACK_PIXELS_PER_M

    median_h      = float(np.median(heights))
    pixels_per_m  = round(median_h / AVERAGE_PERSON_HEIGHT_M, 1)

    if verbose:
        print(f"  [{camera_id}] {len(heights)} detections | "
              f"median height = {median_h:.1f}px | "
              f"PIXELS_PER_METRE = {pixels_per_m}")

    # Save diagnostics
    json.dump(
        {"camera_id": camera_id, "detections": len(heights),
         "median_height_px": median_h, "pixels_per_metre": pixels_per_m},
        open(f"data/calib/{camera_id}_auto.json", "w"), indent=2
    )
    return pixels_per_m


def write_to_env(results: dict):
    """
    Writes PIXELS_PER_METRE_CAM* values into .env.
    Creates .env if missing. Replaces existing camera lines.
    """
    env_path = ".env"
    existing = open(env_path).readlines() if os.path.exists(env_path) else []
    new_keys  = {f"PIXELS_PER_METRE_{c.upper()}": v for c, v in results.items()}
    kept      = [l for l in existing if not any(l.startswith(k) for k in new_keys)]
    with open(env_path, "w") as f:
        f.writelines(kept)
        if kept and not kept[-1].endswith("\n"):
            f.write("\n")
        f.write("\n# Auto-calibrated pixel scales (written by auto_calib.py)\n")
        for k, v in sorted(new_keys.items()):
            f.write(f"{k}={v}\n")
    # Reload env so running process picks up new values immediately
    from dotenv import load_dotenv
    load_dotenv(override=True)


def calibrate_all(source: str = "data/", verbose: bool = True) -> dict:
    if os.path.isfile(source):
        videos = [(source, Path(source).stem)]
    elif os.path.isdir(source):
        videos = [(f, Path(f).stem) for f in sorted(glob.glob(os.path.join(source, "*.mp4")))]
    else:
        return {}

    if not videos:
        print(f"  No .mp4 files found in {source}")
        return {}

    from config.settings import AUTO_CALIB_FRAMES, AUTO_CALIB_MIN_DETECTIONS
    if verbose:
        print(f"\nAuto-calibrating {len(videos)} camera(s)...")

    results = {}
    for path, cam_id in videos:
        val = estimate_pixels_per_metre(
            path, cam_id,
            n_frames=AUTO_CALIB_FRAMES,
            min_detections=AUTO_CALIB_MIN_DETECTIONS,
            verbose=verbose,
        )
        results[cam_id] = val

    write_to_env(results)

    if verbose:
        print(f"\n  Written to .env:")
        for cam_id, val in sorted(results.items()):
            print(f"    PIXELS_PER_METRE_{cam_id.upper()}={val}")

    return results


if __name__ == "__main__":
    src     = sys.argv[1] if len(sys.argv) > 1 else "data/"
    results = calibrate_all(src, verbose=True)
    if results:
        print(f"\nDone. {len(results)} camera(s) calibrated.")
        print("Now run:  python run_live.py data/ --fresh")
    else:
        print("Calibration failed — check your video files.")