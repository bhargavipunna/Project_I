"""
core/interaction/interaction_detector.py
=========================================
WHAT CHANGED FROM ORIGINAL:
  1. Per-camera PIXELS_PER_METRE via get_pixels_per_metre(cam_id)
     — was using single hardcoded fallback for all cameras
     — your overhead cameras need ~652, not 100 — this was the 1-incident bug

  2. Homography distance when calib.py has been run
     — warps pixel coords to real floor metres — much more accurate than flat scale

  3. event_fired → refire every INTERACTION_REFIRE_S
     — was firing ONCE then ignoring rest of conversation
     — now refires every 30s so long interactions grow confidence correctly

  4. get_nearby_count() for accurate group size
     — was passing len(all_persons_in_frame) to classifier
     — now only counts people within 2x radius of the interacting pair

  5. DEBUG_DISTANCES mode from settings
     — prints raw pixel + metre distances per pair per frame
     — set DEBUG_DISTANCES=true in .env to diagnose distance issues

  6. INTERACTION_DISTANCE_M, INTERACTION_DURATION_S, INTERACTION_REFIRE_S
     — all read from settings (env) not hardcoded
"""

import os, time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from loguru import logger
import numpy as np
import cv2

from config.settings import (
    INTERACTION_DISTANCE_M,
    INTERACTION_DURATION_S,
    INTERACTION_REFIRE_S,
    DEBUG_DISTANCES,
    get_pixels_per_metre,
    get_homography,
)


@dataclass
class InteractionEvent:
    person_id_a : str
    person_id_b : str
    camera_id   : str
    frame_num   : int
    start_time  : float
    duration_s  : float
    distance_m  : float
    is_new_pair : bool = False
    is_refire   : bool = False

    @property
    def pair_key(self) -> str:
        ids = sorted([self.person_id_a, self.person_id_b])
        return f"{ids[0]}::{ids[1]}"

    def __str__(self):
        tag = " [REFIRE]" if self.is_refire else ""
        return (
            f"INTERACTION{tag} | {self.person_id_a} ↔ {self.person_id_b} | "
            f"cam={self.camera_id} | dur={self.duration_s:.1f}s | "
            f"dist={self.distance_m:.2f}m | "
            f"{'NEW PAIR' if self.is_new_pair else 'repeat'}"
        )


@dataclass
class ProximityState:
    person_id_a    : str
    person_id_b    : str
    start_time     : float = field(default_factory=time.time)
    start_frame    : int   = 0
    last_frame     : int   = 0
    event_fired    : bool  = False
    last_event_time: float = 0.0
    current_dist_m : float = 0.0


class InteractionDetector:
    """
    Per-camera interaction detector.
    Create one per camera — already done in run_live.py via defaultdict(InteractionDetector).
    Now call as InteractionDetector(camera_id="cam1") so it loads the right scale.
    """

    def __init__(
        self,
        camera_id  : str   = "cam1",
        distance_m : float = INTERACTION_DISTANCE_M,
        duration_s : float = INTERACTION_DURATION_S,
        refire_s   : float = INTERACTION_REFIRE_S,
    ):
        self.camera_id    = camera_id
        self.distance_m   = distance_m
        self.duration_s   = duration_s
        self.refire_s     = refire_s

        # Load per-camera pixel scale (written by auto_calib.py)
        self.pixels_per_m = get_pixels_per_metre(camera_id)

        # Load homography matrix if calib.py was run for this camera
        self.H = get_homography(camera_id)

        if self.H is not None:
            logger.info(f"[{camera_id}] Using homography matrix for distance")
        else:
            logger.info(
                f"[{camera_id}] Using flat scale: {self.pixels_per_m:.1f} px/m "
                f"(run calib.py to update)"
            )

        self._active  : Dict[str, ProximityState] = {}
        self._known   : Set[str]                  = set()
        self._history : List[InteractionEvent]    = []

    # ── Public ────────────────────────────────────────────────────────────────

    def update(
        self,
        persons   : List[Tuple[str, Tuple[int, int]]],
        camera_id : str,
        frame_num : int,
    ) -> List[InteractionEvent]:

        new_events: List[InteractionEvent] = []

        if len(persons) < 2:
            self._expire_missing(set(), frame_num)
            return []

        active_this_frame: Set[str] = set()

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                pid_a, ca = persons[i]
                pid_b, cb = persons[j]

                if pid_a == pid_b:
                    continue

                pair_key = self._pair_key(pid_a, pid_b)
                dist_m   = self._compute_distance_m(ca, cb)

                if DEBUG_DISTANCES:
                    px = self._pixel_distance(ca, cb)
                    status = "IN RANGE" if dist_m < self.distance_m else "too far"
                    print(
                        f"[DIST][{camera_id}] P{pid_a}{ca} <-> P{pid_b}{cb}  "
                        f"px={px:.1f}  m={dist_m:.2f}  [{status}]"
                    )

                if dist_m < self.distance_m:
                    active_this_frame.add(pair_key)

                    if pair_key not in self._active:
                        self._active[pair_key] = ProximityState(
                            person_id_a=pid_a, person_id_b=pid_b,
                            start_time=time.time(), start_frame=frame_num,
                            last_frame=frame_num, current_dist_m=dist_m,
                        )
                    else:
                        state = self._active[pair_key]
                        state.last_frame     = frame_num
                        state.current_dist_m = dist_m
                        duration             = time.time() - state.start_time
                        since_last           = time.time() - state.last_event_time

                        # Initial event when duration threshold first crossed
                        if duration >= self.duration_s and not state.event_fired:
                            state.event_fired    = True
                            state.last_event_time = time.time()
                            is_new = pair_key not in self._known
                            self._known.add(pair_key)
                            ev = InteractionEvent(
                                person_id_a=pid_a, person_id_b=pid_b,
                                camera_id=camera_id, frame_num=frame_num,
                                start_time=state.start_time, duration_s=duration,
                                distance_m=dist_m, is_new_pair=is_new, is_refire=False,
                            )
                            self._history.append(ev)
                            new_events.append(ev)
                            logger.info(str(ev))

                        # Refire every REFIRE_S during sustained interaction
                        # This is what grows confidence for long conversations
                        elif state.event_fired and self.refire_s > 0 and since_last >= self.refire_s:
                            state.last_event_time = time.time()
                            ev = InteractionEvent(
                                person_id_a=pid_a, person_id_b=pid_b,
                                camera_id=camera_id, frame_num=frame_num,
                                start_time=state.start_time, duration_s=duration,
                                distance_m=dist_m, is_new_pair=False, is_refire=True,
                            )
                            self._history.append(ev)
                            new_events.append(ev)
                            logger.info(str(ev))

        self._expire_missing(active_this_frame, frame_num)
        return new_events

    def get_nearby_count(
        self,
        center_a    : Tuple[int, int],
        center_b    : Tuple[int, int],
        all_persons : List[Tuple[str, Tuple[int, int]]],
        multiplier  : float = 2.0,
    ) -> int:
        """
        Count only people near the interacting pair, not all people in frame.
        Fixes GROUP_GATHERING being triggered when 20 strangers share a wide frame.
        """
        mid = ((center_a[0] + center_b[0]) // 2, (center_a[1] + center_b[1]) // 2)
        return sum(
            1 for _, c in all_persons
            if self._compute_distance_m(mid, c) < self.distance_m * multiplier
        )

    def get_event_history(self)        -> List[InteractionEvent]: return list(self._history)
    def get_active_proximities(self)   -> Dict[str, ProximityState]: return dict(self._active)
    def get_interaction_count(self)    -> int: return len(self._history)

    def reset(self):
        self._active.clear(); self._known.clear(); self._history.clear()

    # ── Distance ──────────────────────────────────────────────────────────────

    def _compute_distance_m(self, ca: Tuple[int,int], cb: Tuple[int,int]) -> float:
        if self.H is not None:
            return self._homography_distance(ca, cb)
        return self._flat_distance(ca, cb)

    def _homography_distance(self, ca, cb) -> float:
        def warp(pt):
            p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
            return cv2.perspectiveTransform(p, self.H)[0][0]
        return float(np.linalg.norm(warp(ca) - warp(cb)))

    def _flat_distance(self, ca, cb) -> float:
        return self._pixel_distance(ca, cb) / self.pixels_per_m

    def _pixel_distance(self, ca, cb) -> float:
        dx, dy = ca[0]-cb[0], ca[1]-cb[1]
        return float(np.sqrt(dx*dx + dy*dy))

    def _pair_key(self, a: str, b: str) -> str:
        return "::".join(sorted([a, b]))

    def _expire_missing(self, active: Set[str], frame_num: int):
        for key in [k for k in self._active if k not in active]:
            state = self._active.pop(key)
            logger.debug(
                f"Proximity ended: {state.person_id_a} ↔ {state.person_id_b} | "
                f"dur={time.time()-state.start_time:.1f}s | fired={state.event_fired}"
            )


def annotate_interactions(frame, persons_with_ids, active_proximities, events_this_frame):
    annotated    = frame.copy()
    centers      = {pid: c for pid, c in persons_with_ids}
    for key, state in active_proximities.items():
        ca, cb = centers.get(state.person_id_a), centers.get(state.person_id_b)
        if ca and cb:
            dur       = time.time() - state.start_time
            intensity = min(255, int(255 * dur / max(INTERACTION_DURATION_S, 0.1)))
            cv2.line(annotated, ca, cb, (0, intensity, intensity), 2)
            mid = ((ca[0]+cb[0])//2, (ca[1]+cb[1])//2)
            cv2.putText(annotated, f"{dur:.1f}s", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,intensity,intensity), 1)
    for ev in events_this_frame:
        ca, cb = centers.get(ev.person_id_a), centers.get(ev.person_id_b)
        if ca and cb:
            cv2.line(annotated, ca, cb, (0,0,255), 3)
            mid = ((ca[0]+cb[0])//2, (ca[1]+cb[1])//2)
            cv2.putText(annotated, "REFIRE" if ev.is_refire else "INTERACTION",
                        (mid[0]-40, mid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
    cv2.putText(annotated, f"Proximities: {len(active_proximities)}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    return annotated