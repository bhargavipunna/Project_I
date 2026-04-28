"""
Bird-eye unified floor map utilities for multi-camera scene understanding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.settings import WORLD_MAP_SCALE_PX_PER_M, get_homography


@dataclass
class FloorPoint:
    person_id: str
    camera_id: str
    world_xy_m: Tuple[float, float]
    map_xy_px: Tuple[int, int]


class UnifiedFloorMapper:
    """Projects camera pixel points to a common world/map coordinate frame."""

    def __init__(self, map_scale_px_per_m: float = WORLD_MAP_SCALE_PX_PER_M):
        self.map_scale = map_scale_px_per_m
        self._homographies: Dict[str, Optional[np.ndarray]] = {}

    def to_world_xy(self, camera_id: str, pixel_xy: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        h = self._homographies.get(camera_id)
        if h is None and camera_id not in self._homographies:
            h = get_homography(camera_id)
            self._homographies[camera_id] = h
        if h is None:
            return None

        p = np.array([[[float(pixel_xy[0]), float(pixel_xy[1])]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(p, h)[0][0]
        return float(warped[0]), float(warped[1])

    def world_to_map(self, world_xy_m: Tuple[float, float]) -> Tuple[int, int]:
        return int(world_xy_m[0] * self.map_scale), int(world_xy_m[1] * self.map_scale)

    def make_point(self, person_id: str, camera_id: str, pixel_xy: Tuple[int, int]) -> Optional[FloorPoint]:
        world_xy = self.to_world_xy(camera_id, pixel_xy)
        if world_xy is None:
            return None
        return FloorPoint(
            person_id=person_id,
            camera_id=camera_id,
            world_xy_m=world_xy,
            map_xy_px=self.world_to_map(world_xy),
        )
