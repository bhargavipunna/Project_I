"""
config/settings.py
==================
WHAT CHANGED:
  Every single hardcoded number is now loaded from .env
  Per-camera helpers replace single global values
  Privacy, agent, and MacBook-CPU settings added
  get_pixels_per_metre(cam_id) replaces single PIXELS_PER_METRE
  get_reid_threshold(cam_id) replaces single REID_THRESHOLD
  get_homography(cam_id) loads saved calib matrix if available
"""

import os, json
from dotenv import load_dotenv
load_dotenv(override=True)

# ── Re-ID ─────────────────────────────────────────────────────────────────────
REID_THRESHOLD = float(os.getenv("FAISS_MATCH_THRESHOLD", "0.65"))
FAISS_MATCH_THRESHOLD = REID_THRESHOLD

def get_reid_threshold(camera_id: str) -> float:
    """Per-camera threshold. Falls back to FAISS_MATCH_THRESHOLD."""
    return float(os.getenv(f"FAISS_THRESHOLD_{camera_id.upper()}", str(REID_THRESHOLD)))

# ── Interaction detection ─────────────────────────────────────────────────────
INTERACTION_DISTANCE_M = float(os.getenv("INTERACTION_DISTANCE_M", "1.5"))
INTERACTION_DURATION_S = float(os.getenv("INTERACTION_DURATION_S", "2.0"))
INTERACTION_REFIRE_S   = float(os.getenv("INTERACTION_REFIRE_S",   "30.0"))

# ── Pixel → metre ─────────────────────────────────────────────────────────────
# FALLBACK only. auto_calib.py writes per-camera values to .env automatically.
PIXELS_PER_METRE = float(os.getenv("PIXELS_PER_METRE", "100.0"))

def get_pixels_per_metre(camera_id: str) -> float:
    """Per-camera scale. Written by auto_calib.py on startup."""
    return float(os.getenv(f"PIXELS_PER_METRE_{camera_id.upper()}", str(PIXELS_PER_METRE)))

def get_homography(camera_id: str):
    """
    Returns 3x3 numpy homography matrix if calib.py was run for this camera.
    Returns None if not available (flat scale used instead).
    """
    import numpy as np
    path = f"data/calib/{camera_id}_H.json"
    if os.path.exists(path):
        with open(path) as f:
            return np.array(json.load(f), dtype=np.float32)
    return None

# ── Confidence engine ─────────────────────────────────────────────────────────
DECAY_RATE                    = float(os.getenv("DECAY_RATE",                    "0.998"))
DECAY_INTERVAL_M              = float(os.getenv("DECAY_INTERVAL_M",              "10"))
DECAY_INTERVAL_MINUTES        = int(os.getenv("DECAY_INTERVAL_MINUTES", str(int(DECAY_INTERVAL_M))))
MIN_CONFIDENCE_FLOOR          = float(os.getenv("MIN_CONFIDENCE_FLOOR",          "0.05"))
MIN_CONFIDENCE_FLOOR_MEETINGS = int(  os.getenv("MIN_CONFIDENCE_FLOOR_MEETINGS", "5"))

# ── Boost modifiers ───────────────────────────────────────────────────────────
MAX_DISTANCE_M          = INTERACTION_DISTANCE_M
LOCATION_BUCKET_PX      = int(  os.getenv("LOCATION_BUCKET_PX",      "100"))
LOCATION_MAX_BONUS      = float(os.getenv("LOCATION_MAX_BONUS",      "0.3"))
LOCATION_VISITS_FOR_MAX = int(  os.getenv("LOCATION_VISITS_FOR_MAX", "5"))
PRIVACY_ONE_ON_ONE      = float(os.getenv("PRIVACY_ONE_ON_ONE",      "1.2"))
PRIVACY_GROUP           = float(os.getenv("PRIVACY_GROUP",           "0.8"))
DIMINISHING_RATE        = float(os.getenv("DIMINISHING_RATE",        "0.3"))
DIMINISHING_WINDOW_H    = float(os.getenv("DIMINISHING_WINDOW_H",    "4.0"))

# ── Privacy (DPDP Act compliant) ──────────────────────────────────────────────
# Auto-delete identities not seen for this many days
IDENTITY_RETENTION_DAYS   = int( os.getenv("IDENTITY_RETENTION_DAYS",   "30"))
# Always strip person IDs before any agent/API call
ANONYMISE_FOR_AGENT       = os.getenv("ANONYMISE_FOR_AGENT", "true").lower() == "true"
# Print raw pixel distances per frame — set true to diagnose 1-incident bug
DEBUG_DISTANCES           = os.getenv("DEBUG_DISTANCES", "false").lower() == "true"

# ── Local Ollama agent (free, fully private) ──────────────────────────────────
# Install:  curl -fsSL https://ollama.com/install.sh | sh
# Pull:     ollama pull llama3.2:3b
OLLAMA_HOST      = os.getenv("OLLAMA_HOST",      "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "llama3.2:3b")
AGENT_ENABLED    = os.getenv("AGENT_ENABLED",    "false").lower() == "true"
AGENT_INTERVAL_M = float(os.getenv("AGENT_INTERVAL_M", "10"))

# ── GPU / Device Auto-detection ─────────────────────────────────────────────
# Apple Silicon → MPS, NVIDIA → CUDA, fallback → CPU
# Override with YOLO_DEVICE env var if needed
def _auto_device() -> str:
    """Auto-detect best available compute device."""
    import torch
    if os.getenv("YOLO_DEVICE"):           # Explicit override
        return os.getenv("YOLO_DEVICE")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    if torch.cuda.is_available():          # NVIDIA GPU
        return "cuda"
    return "cpu"

# Process every Nth frame — 2 is good on GPU, 4 on CPU
FRAME_SKIP       = int(  os.getenv("FRAME_SKIP",       "2"))
# Resize before detection
DETECTION_WIDTH  = int(  os.getenv("DETECTION_WIDTH",  "640"))
DETECTION_HEIGHT = int(  os.getenv("DETECTION_HEIGHT", "480"))
# yolov8s = small model, much better than nano on GPU
YOLO_MODEL       = os.getenv("YOLO_MODEL",      "yolov8s.pt")
YOLO_CONFIDENCE  = float(os.getenv("YOLO_CONFIDENCE",  "0.45"))
YOLO_DEVICE      = _auto_device()
BOTSORT_BUFFER_FRAMES = int(os.getenv("BOTSORT_BUFFER_FRAMES", "90"))


# Re-ID model defaults (used by embedder/identity manager modules)
OSNET_MODEL      = os.getenv("OSNET_MODEL", "osnet_x0_25")
EMBEDDING_DIM    = int(os.getenv("EMBEDDING_DIM", "512"))
ENABLE_GAIT_FUSION = os.getenv("ENABLE_GAIT_FUSION", "false").lower() == "true"
GAIT_FUSION_WEIGHT = float(os.getenv("GAIT_FUSION_WEIGHT", "0.35"))
GAIT_EMBEDDING_DIM = int(os.getenv("GAIT_EMBEDDING_DIM", "128"))
GAIT_MODEL_PATH    = os.getenv("GAIT_MODEL_PATH", "")

# ── Bird-eye / unified floor map ───────────────────────────────────────────────
ENABLE_UNIFIED_FLOOR_MAP = os.getenv("ENABLE_UNIFIED_FLOOR_MAP", "true").lower() == "true"
WORLD_MAP_SCALE_PX_PER_M = float(os.getenv("WORLD_MAP_SCALE_PX_PER_M", "30.0"))

# Backward-compatible aliases for modules using old names
FRAME_WIDTH      = DETECTION_WIDTH
FRAME_HEIGHT     = DETECTION_HEIGHT
VIDEO_SOURCE     = os.getenv("VIDEO_SOURCE", "0")
RTSP_RECONNECT_DELAY = int(os.getenv("RTSP_RECONNECT_DELAY", "3"))

# ── Auto-calibration ──────────────────────────────────────────────────────────
AUTO_CALIB_ON_STARTUP     = os.getenv("AUTO_CALIB_ON_STARTUP", "false").lower() == "true"
AUTO_CALIB_MIN_DETECTIONS = int(os.getenv("AUTO_CALIB_MIN_DETECTIONS", "8"))
AUTO_CALIB_FRAMES         = int(os.getenv("AUTO_CALIB_FRAMES",         "150"))

# ── Paths ─────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH    = os.getenv("FAISS_INDEX_PATH",    "data/embeddings/faiss.index")
IDENTITY_MAP_PATH   = os.getenv("IDENTITY_MAP_PATH",   "data/embeddings/identity_map.json")
GRAPH_SNAPSHOT_PATH = os.getenv("GRAPH_SNAPSHOT_PATH", "data/snapshots/graph.json")
CALIB_DIR           = os.getenv("CALIB_DIR",           "data/calib")

# ── Graph persistence / analytics ─────────────────────────────────────────────
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "networkx").lower()  # networkx|neo4j
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
ENABLE_LOUVAIN_COMMUNITIES = os.getenv("ENABLE_LOUVAIN_COMMUNITIES", "true").lower() == "true"

# ── Redis Streams workers ─────────────────────────────────────────────────────
ENABLE_REDIS_STREAMS = os.getenv("ENABLE_REDIS_STREAMS", "false").lower() == "true"
REDIS_URL             = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_STREAM_INCIDENTS = os.getenv("REDIS_STREAM_INCIDENTS", "urgis:incidents")
REDIS_STREAM_MAXLEN    = int(os.getenv("REDIS_STREAM_MAXLEN", "10000"))
REDIS_CONSUMER_GROUP   = os.getenv("REDIS_CONSUMER_GROUP", "urgis-workers")