"""
pipeline_state.py
=================
Singleton module for URG-IS pipeline state.

WHY A SEPARATE MODULE?
Streamlit re-executes mvp_dashboard.py on every page refresh.
Any dict/var defined at module-level in the dashboard gets RESET each rerun.
Python caches imported modules in sys.modules — so state stored HERE
persists across all reruns for the lifetime of the server process.
"""

import threading
from collections import deque

# ── Shared state ──────────────────────────────────────────────────────────────
lock = threading.Lock()

G = {
    "running":        False,
    "stop_requested": False,
    "graph_db":       None,
    "engine":         None,
    "incidents":      deque(maxlen=2000),
    "conf_history":   deque(maxlen=5000),
    "latest_frames":  {},   # cam_id -> RGB numpy array
    "frame_counts":   {},   # cam_id -> int
    "person_count":   0,
    "active_cams":    [],
    "last_update":    0.0,
    "total_frames":   0,
    "camera_sources": {},
    "fps":            {},   # cam_id -> float fps
    "errors":         [],
}

pipeline_thread = None   # set by _start_pipeline()
