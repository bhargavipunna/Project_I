"""
Appearance + gait fusion for clothing-change Re-ID.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from config.settings import ENABLE_GAIT_FUSION, GAIT_FUSION_WEIGHT
from core.reid.embedder import PersonEmbedder
from core.reid.gait_embedder import GaitEmbedder


class FusionEmbedder:
    """
    Produces a fused descriptor:
      fused = (1-w)*appearance + w*gait_projected
    """

    def __init__(self, enable_gait_fusion: bool = ENABLE_GAIT_FUSION, gait_weight: float = GAIT_FUSION_WEIGHT):
        self.enable_gait_fusion = enable_gait_fusion
        self.gait_weight = max(0.0, min(0.95, gait_weight))
        self.appearance = PersonEmbedder()
        self.gait = GaitEmbedder() if enable_gait_fusion else None
        logger.info(
            f"FusionEmbedder ready | gait_fusion={self.enable_gait_fusion} | "
            f"gait_weight={self.gait_weight:.2f}"
        )

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        app = self.appearance.embed(crop)
        if app is None:
            return None
        if not self.enable_gait_fusion or self.gait is None:
            return app

        gait_vec = self.gait.embed(crop)
        if gait_vec is None:
            return app

        # Project gait vector to appearance dimensionality via linear interpolation.
        target_dim = app.shape[0]
        gait_resized = np.interp(
            np.linspace(0, len(gait_vec) - 1, target_dim),
            np.arange(len(gait_vec)),
            gait_vec,
        ).astype(np.float32)
        gait_norm = np.linalg.norm(gait_resized)
        if gait_norm > 1e-8:
            gait_resized /= gait_norm

        fused = (1.0 - self.gait_weight) * app + self.gait_weight * gait_resized
        fused_norm = np.linalg.norm(fused)
        if fused_norm > 1e-8:
            fused = fused / fused_norm
        return fused.astype(np.float32)
