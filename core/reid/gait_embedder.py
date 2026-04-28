"""
Optional gait embedding for clothing-change robust Re-ID.

This module is intentionally lightweight:
- If OpenGait artifacts are available, it can be wired here.
- Otherwise it falls back to a silhouette-shape descriptor so fusion still runs.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from loguru import logger

from config.settings import GAIT_EMBEDDING_DIM


class GaitEmbedder:
    """Extracts a compact gait embedding from a person crop."""

    def __init__(self, embedding_dim: int = GAIT_EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self._backend = "silhouette_fallback"
        logger.info(f"GaitEmbedder active | backend={self._backend} | dim={embedding_dim}")

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        if crop.shape[0] < 12 or crop.shape[1] < 12:
            return None

        # Foreground silhouette estimate from crop.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.resize(mask, (64, 128), interpolation=cv2.INTER_AREA)

        # Shape histogram acts as a stable clothing-agnostic fallback proxy.
        col_hist = mask.mean(axis=0).astype(np.float32) / 255.0
        row_hist = mask.mean(axis=1).astype(np.float32) / 255.0
        vec = np.concatenate([col_hist, row_hist], axis=0)

        # Resample to configured dimensionality.
        vec = cv2.resize(vec.reshape(1, -1), (self.embedding_dim, 1), interpolation=cv2.INTER_LINEAR)
        vec = vec.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec
