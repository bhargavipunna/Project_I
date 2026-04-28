"""
core/graph/incident_classifier.py

Classifies every raw interaction event into a typed Incident.
An incident is the fundamental unit of relationship building.

Classification logic:
  Distance and duration together determine the incident type.
  More serious incidents (CLOSE_CONTACT, EXTENDED_MEETING) carry
  higher base confidence boosts than casual ones (PROXIMITY).

Incident types (in order of relationship significance):
  CLOSE_CONTACT     → very close (<0.5m), any duration  — hug/handshake zone
  EXTENDED_MEETING  → close (<1.5m), very long (>120s)  — long discussion
  CONVERSATION      → close (<1.2m), long (>30s)        — talking
  GROUP_GATHERING   → 3+ people within 2m               — group context
  PROXIMITY         → close (<1.5m), brief (2–30s)      — passing/brief stop
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
import time
from loguru import logger


# ── Incident types ────────────────────────────────────────────────────────────

class IncidentType(str, Enum):
    CLOSE_CONTACT    = "CLOSE_CONTACT"     # <0.5m — physical contact zone
    EXTENDED_MEETING = "EXTENDED_MEETING"  # <1.5m, >120s — long deliberate meeting
    CONVERSATION     = "CONVERSATION"      # <1.2m, >30s  — talking
    GROUP_GATHERING  = "GROUP_GATHERING"   # 3+ people within 2m
    PROXIMITY        = "PROXIMITY"         # <1.5m, 2–30s — passing/brief


# Base confidence boost per incident type
INCIDENT_BOOST = {
    IncidentType.CLOSE_CONTACT    : 0.20,
    IncidentType.EXTENDED_MEETING : 0.18,
    IncidentType.CONVERSATION     : 0.12,
    IncidentType.GROUP_GATHERING  : 0.06,
    IncidentType.PROXIMITY        : 0.03,
}

# Human-readable descriptions for dashboard
INCIDENT_DESCRIPTION = {
    IncidentType.CLOSE_CONTACT    : "Physical contact zone (<0.5m)",
    IncidentType.EXTENDED_MEETING : "Extended meeting (>2 min)",
    IncidentType.CONVERSATION     : "Conversation (>30s)",
    IncidentType.GROUP_GATHERING  : "Group gathering (3+ people)",
    IncidentType.PROXIMITY        : "Proximity / passing (<30s)",
}


# ── Incident data ─────────────────────────────────────────────────────────────

@dataclass
class Incident:
    """
    One classified interaction between two people.

    person_id_a    : first person
    person_id_b    : second person
    incident_type  : classified type
    base_boost     : confidence boost before modifiers
    distance_m     : distance in metres when detected
    duration_s     : how long they were close
    camera_id      : which camera
    frame_num      : frame when threshold crossed
    timestamp      : wall-clock time
    location_px    : pixel location of midpoint (for location tracking)
    group_size     : number of people in scene (for group detection)
    is_one_on_one  : True if only 2 people in scene
    """
    person_id_a   : str
    person_id_b   : str
    incident_type : IncidentType
    base_boost    : float
    distance_m    : float
    duration_s    : float
    camera_id     : str
    frame_num     : int
    timestamp     : float = field(default_factory=time.time)
    location_px   : Tuple[int, int] = (0, 0)
    group_size    : int  = 2
    is_one_on_one : bool = True

    @property
    def pair_key(self) -> str:
        return "::".join(sorted([self.person_id_a, self.person_id_b]))

    def __str__(self):
        return (
            f"[{self.incident_type}] "
            f"{self.person_id_a[-5:]} ↔ {self.person_id_b[-5:]} | "
            f"dist={self.distance_m:.2f}m | "
            f"dur={self.duration_s:.1f}s | "
            f"boost={self.base_boost:.2f} | "
            f"cam={self.camera_id}"
        )


# ── Incident Classifier ───────────────────────────────────────────────────────

class IncidentClassifier:
    """
    Converts a raw interaction (distance + duration + context) into a typed Incident.

    Usage:
        classifier = IncidentClassifier()
        incident   = classifier.classify(
            person_id_a  = "PERSON_00001",
            person_id_b  = "PERSON_00002",
            distance_m   = 0.8,
            duration_s   = 45.0,
            camera_id    = "cam1",
            frame_num    = 120,
            location_px  = (320, 240),
            people_in_scene = ["PERSON_00001", "PERSON_00002"],
        )
        print(incident.incident_type)   # CONVERSATION
        print(incident.base_boost)      # 0.12
    """

    # Thresholds
    CLOSE_CONTACT_DISTANCE_M    = 0.5
    CONVERSATION_DISTANCE_M     = 1.2
    PROXIMITY_DISTANCE_M        = 1.5
    EXTENDED_MEETING_DURATION_S = 120.0
    CONVERSATION_DURATION_S     = 30.0
    GROUP_MIN_PEOPLE            = 3
    GROUP_GATHERING_MIN_DURATION_S = 5.0
    GROUP_GATHERING_MAX_DISTANCE_M = 1.2

    def classify(
        self,
        person_id_a      : str,
        person_id_b      : str,
        distance_m       : float,
        duration_s       : float,
        camera_id        : str,
        frame_num        : int,
        location_px      : Tuple[int, int] = (0, 0),
        people_in_scene  : Optional[List[str]] = None,
        timestamp        : Optional[float] = None,
    ) -> Incident:
        """
        Classify one interaction into an Incident.
        Rules applied in order of priority (most significant first).
        """
        if people_in_scene is None:
            people_in_scene = [person_id_a, person_id_b]

        group_size    = len(people_in_scene)
        is_one_on_one = group_size == 2

        # ── Classification rules (priority order) ──────────────────────────
        if distance_m < self.CLOSE_CONTACT_DISTANCE_M:
            incident_type = IncidentType.CLOSE_CONTACT

        elif duration_s >= self.EXTENDED_MEETING_DURATION_S:
            incident_type = IncidentType.EXTENDED_MEETING

        elif (duration_s >= self.CONVERSATION_DURATION_S and
              distance_m < self.CONVERSATION_DISTANCE_M):
            incident_type = IncidentType.CONVERSATION

        elif (
            group_size >= self.GROUP_MIN_PEOPLE
            and duration_s >= self.GROUP_GATHERING_MIN_DURATION_S
            and distance_m < self.GROUP_GATHERING_MAX_DISTANCE_M
        ):
            # Suppress noisy group labels from very brief/weak crowd proximity.
            incident_type = IncidentType.GROUP_GATHERING

        else:
            incident_type = IncidentType.PROXIMITY

        incident = Incident(
            person_id_a   = person_id_a,
            person_id_b   = person_id_b,
            incident_type = incident_type,
            base_boost    = INCIDENT_BOOST[incident_type],
            distance_m    = distance_m,
            duration_s    = duration_s,
            camera_id     = camera_id,
            frame_num     = frame_num,
            timestamp     = timestamp or time.time(),
            location_px   = location_px,
            group_size    = group_size,
            is_one_on_one = is_one_on_one,
        )

        logger.debug(str(incident))
        return incident