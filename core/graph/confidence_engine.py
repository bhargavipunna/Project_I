"""
core/graph/confidence_engine.py
STEP 6b — Confidence Engine (Incident-Based Multi-Factor Model)

Converts a classified Incident into a confidence score using 4 modifiers:

  Base boost (from incident type):
    PROXIMITY         → 0.03
    CONVERSATION      → 0.12
    CLOSE_CONTACT     → 0.20
    EXTENDED_MEETING  → 0.18
    GROUP_GATHERING   → 0.06

  Modifier 1 — Distance (closer = stronger):
    multiplier = 1.0 + (max_distance - distance_m) / max_distance
    At 0.1m → 1.93×   At 0.75m → 1.5×   At 1.5m → 1.0×

  Modifier 2 — Location (same spot repeatedly = routine):
    multiplier = 1.0 + 0.3 × min(same_location_visits, 5) / 5
    5+ visits at same spot → 1.30×

  Modifier 3 — Privacy (one-on-one vs group):
    one-on-one → 1.2×   group → 0.8×

  Modifier 4 — Diminishing returns (same pair, same day):
    multiplier = 1 / (1 + meetings_today × 0.3)
    Meeting 1 → 1.00×   Meeting 2 → 0.77×   Meeting 5 → 0.45×

  Final boost = base × dist × location × privacy × diminishing
  New confidence = min(1.0, old_confidence + final_boost)

  Decay: multiplicative every 10 min
    confidence × 0.998 per tick
    Strong (0.80) → 27 hrs to halve
    Weak   (0.20) →  6 hrs to halve
"""

import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from apscheduler.schedulers.background import BackgroundScheduler

from config.settings import DECAY_INTERVAL_MINUTES
from core.graph.graph_db import GraphDB, RelationshipEdge, relationship_label
from core.graph.incident_classifier import (
    Incident, IncidentType, IncidentClassifier, INCIDENT_BOOST
)
from core.interaction.interaction_detector import InteractionEvent


import time
from config.settings import (
    DECAY_RATE,
    MIN_CONFIDENCE_FLOOR,
    MIN_CONFIDENCE_FLOOR_MEETINGS,
    DIMINISHING_RATE,
    DIMINISHING_WINDOW_H,
    MAX_DISTANCE_M,
    LOCATION_BUCKET_PX,
    LOCATION_MAX_BONUS,
    LOCATION_VISITS_FOR_MAX,
    PRIVACY_ONE_ON_ONE,
    PRIVACY_GROUP,
)



# ── Per-pair modifier state ───────────────────────────────────────────────────

@dataclass
class PairState:
    """Tracks modifier-relevant state for one pair of people."""
    # Location visit counts: bucketed_location_key → visit_count
    location_visits : Dict[str, int] = field(default_factory=dict)
    # Daily meeting tracking
    meetings_today  : int   = 0
    last_day_key    : str   = ""
    # Full confidence history for audit
    confidence_log  : list  = field(default_factory=list)  # [(timestamp, confidence, incident_type)]


# ── Confidence Engine ─────────────────────────────────────────────────────────

class ConfidenceEngine:
    """
    Incident-based confidence scoring with 4 modifiers.

    Usage:
        db         = GraphDB()
        engine     = ConfidenceEngine(db)
        classifier = IncidentClassifier()
        engine.start()

        # From an interaction event:
        incident = classifier.classify(...)
        result   = engine.process_incident(incident)
        print(result.confidence, result.relationship)

        # Per-person graph:
        graph = engine.get_person_graph("PERSON_00001")

        engine.stop()
    """

    def __init__(
        self,
        graph_db         : GraphDB,
        decay_interval_m : int  = DECAY_INTERVAL_MINUTES,
        auto_snapshot    : bool = True,
    ):
        self._db          = graph_db
        self._classifier  = IncidentClassifier()
        self._scheduler   = BackgroundScheduler()
        self._running     = False
        self._auto_snap   = auto_snapshot
        self._decay_interval = decay_interval_m

        # Per-pair modifier state
        self._pair_states : Dict[str, PairState] = {}

        self._incident_count = 0
        self._decay_count    = 0

        logger.info(
            f"ConfidenceEngine ready | "
            f"decay×{DECAY_RATE}/tick every {decay_interval_m}min"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._scheduler.add_job(
            func    = self._decay_job,
            trigger = "interval",
            minutes = self._decay_interval,
            id      = "decay_job",
        )
        self._scheduler.start()
        self._running = True
        logger.success(f"Decay scheduler started — every {self._decay_interval} min")

    def stop(self):
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
        if self._auto_snap:
            self._db.save_snapshot()
        logger.info(f"ConfidenceEngine stopped | incidents={self._incident_count}")

    # ── Main processing ───────────────────────────────────────────────────────

    def process_event(
        self,
        event            : InteractionEvent,
        people_in_scene  : Optional[List[str]] = None,
    ) -> RelationshipEdge:
        """
        Full pipeline:
          1. Classify interaction event → Incident
          2. Compute all modifiers
          3. Calculate new confidence
          4. Write to graph
          5. Return updated edge
        """
        # Step 1 — classify
        incident = self._classifier.classify(
            person_id_a     = event.person_id_a,
            person_id_b     = event.person_id_b,
            distance_m      = event.distance_m,
            duration_s      = event.duration_s,
            camera_id       = event.camera_id,
            frame_num       = event.frame_num,
            location_px     = self._midpoint(event),
            people_in_scene = people_in_scene or [event.person_id_a, event.person_id_b],
            timestamp       = event.start_time,
        )

        # Step 2 — compute modifiers
        boost, breakdown = self._compute_boost(incident)

        # Step 3 — get current confidence and add boost
        existing = self._db.get_edge(incident.person_id_a, incident.person_id_b)
        current_conf = existing.confidence if existing else 0.0
        new_conf = round(min(1.0, current_conf + boost), 4)

        # Step 4 — write to graph
        edge = self._db.record_incident(incident, confidence=new_conf)

        # Step 5 — log to pair state
        state = self._get_state(incident.pair_key)
        state.confidence_log.append((
            incident.timestamp,
            new_conf,
            incident.incident_type.value,
        ))

        self._incident_count += 1

        logger.info(
            f"[{incident.incident_type.value}] "
            f"{incident.person_id_a[-5:]} ↔ {incident.person_id_b[-5:]} | "
            f"conf {current_conf:.3f} → {new_conf:.3f} ({edge.relationship}) | "
            f"boost={boost:.4f} "
            f"[base={breakdown['base']:.3f} "
            f"dist×{breakdown['dist_mod']:.2f} "
            f"loc×{breakdown['loc_mod']:.2f} "
            f"priv×{breakdown['priv_mod']:.2f} "
            f"dim×{breakdown['dim_mod']:.2f}]"
        )

        return edge

    def process_events(
        self,
        events          : List[InteractionEvent],
        people_in_scene : Optional[List[str]] = None,
    ) -> List[RelationshipEdge]:
        return [self.process_event(e, people_in_scene) for e in events]

    # ── Modifier computation ──────────────────────────────────────────────────

    def _compute_boost(self, incident: Incident) -> Tuple[float, Dict]:
        """
        Applies all 4 modifiers to the base boost.
        Returns (final_boost, breakdown_dict).
        """
        base      = incident.base_boost
        dist_mod  = self._distance_modifier(incident.distance_m)
        loc_mod   = self._location_modifier(incident)
        priv_mod  = PRIVACY_ONE_ON_ONE if incident.is_one_on_one else PRIVACY_GROUP
        dim_mod   = self._diminishing_modifier(incident)

        final = base * dist_mod * loc_mod * priv_mod * dim_mod
        final = round(max(0.001, final), 6)

        return final, {
            "base"    : base,
            "dist_mod": round(dist_mod, 3),
            "loc_mod" : round(loc_mod, 3),
            "priv_mod": round(priv_mod, 3),
            "dim_mod" : round(dim_mod, 3),
            "final"   : round(final, 4),
        }

    def _distance_modifier(self, distance_m: float) -> float:
        """
        Closer = stronger signal.
        At 0.1m → 1.93×   At 0.75m → 1.5×   At 1.5m → 1.0×
        """
        distance_m = max(0.05, distance_m)
        return 1.0 + (MAX_DISTANCE_M - distance_m) / MAX_DISTANCE_M

    def _location_modifier(self, incident: Incident) -> float:
        """
        Same location repeatedly = routine meeting = stronger signal.
        Buckets locations into grid cells of LOCATION_BUCKET_PX.
        """
        state   = self._get_state(incident.pair_key)
        loc_key = self._bucket_location(incident.location_px)

        state.location_visits[loc_key] = state.location_visits.get(loc_key, 0) + 1
        visits = state.location_visits[loc_key]

        bonus = LOCATION_MAX_BONUS * min(visits, LOCATION_VISITS_FOR_MAX) / LOCATION_VISITS_FOR_MAX
        return round(1.0 + bonus, 3)

    def _diminishing_modifier(self, incident: Incident) -> float:
        """
        Reduce repeated boosts for the same pair within a day.
        Meeting #1 -> 1.00, Meeting #2 -> 1/(1+0.3), ...
        """
        state = self._get_state(incident.pair_key)
        day_key = self._day_key()
        if state.last_day_key != day_key:
            state.last_day_key = day_key
            state.meetings_today = 0

        mod = 1.0 / (1.0 + state.meetings_today * DIMINISHING_RATE)
        state.meetings_today += 1
        return round(mod, 4)

    def get_diminishing_mod_PATCHED(pair_key: str, meeting_timestamps: dict) -> float:
        """
        Diminishing returns based on meetings in last DIMINISHING_WINDOW_H hours.
        Replaces meetings_today counter which reset at midnight (timezone bug).
    
        meeting_timestamps: dict mapping pair_key -> list of unix timestamps
        Add: meeting_timestamps[pair_key].append(time.time()) on each interaction.
        """
        window_cutoff = time.time() - (DIMINISHING_WINDOW_H * 3600)
        recent = sum(1 for t in meeting_timestamps.get(pair_key, []) if t > window_cutoff)
        return 1.0 / (1.0 + recent * DIMINISHING_RATE)
 
    def purge_old_identities_PATCHED(self):
        """
        Delete identities not seen for IDENTITY_RETENTION_DAYS.
        Satisfies DPDP Act right-to-erasure by default.
        Call from APScheduler alongside the decay job.
        """
        from config.settings import IDENTITY_RETENTION_DAYS
        import os
    
        cutoff   = time.time() - (IDENTITY_RETENTION_DAYS * 86400)
        all_nodes = self.db.get_all_nodes()
    
        for node in all_nodes:
            last_seen = getattr(node, "last_seen", None)
            if last_seen and last_seen < cutoff:
                # Remove from graph
                self.db.delete_person(node.person_id)
                # Remove from FAISS if identity_manager is accessible
                # identity_manager.remove(node.person_id)  # add this if you have the ref
                from loguru import logger
                logger.info(f"Privacy purge: removed {node.person_id} "
                            f"(last seen {(time.time()-last_seen)/86400:.1f} days ago)")
    
 
    # ── Per-person graph queries ──────────────────────────────────────────────

    def get_person_graph(self, person_id: str) -> Optional[Dict]:
        """Returns personal relationship graph for one person."""
        return self._db.get_person_graph(person_id)

    def get_all_person_graphs(self) -> Dict[str, Dict]:
        """Returns personal graphs for all known people."""
        return self._db.get_all_person_graphs()

    def get_confidence_log(self, pid_a: str, pid_b: str) -> list:
        """Returns full confidence history for a pair [(timestamp, confidence, incident_type)]."""
        key = "::".join(sorted([pid_a, pid_b]))
        state = self._pair_states.get(key)
        return state.confidence_log if state else []

    def get_stats(self) -> dict:
        return {
            "nodes"          : self._db.get_node_count(),
            "edges"          : self._db.get_edge_count(),
            "incidents_total": self._incident_count,
            "decay_runs"     : self._decay_count,
            "running"        : self._running,
        }

    def force_decay(self) -> int:
        return self._decay_job()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_state(self, pair_key: str) -> PairState:
        if pair_key not in self._pair_states:
            self._pair_states[pair_key] = PairState()
        return self._pair_states[pair_key]

    def _bucket_location(self, loc_px: Tuple[int, int]) -> str:
        bx = loc_px[0] // LOCATION_BUCKET_PX
        by = loc_px[1] // LOCATION_BUCKET_PX
        return f"{bx}:{by}"

    def _midpoint(self, event: InteractionEvent) -> Tuple[int, int]:
        return (0, 0)   # pipeline.py fills this in when it has centers

    def _day_key(self) -> str:
        t = time.localtime()
        return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"

    def _decay_job(self) -> int:
        logger.info(f"Decay tick | multiplier={DECAY_RATE}")
        deleted = self._db.apply_decay(DECAY_RATE)
        self._decay_count += 1
        if self._auto_snap:
            self._db.save_snapshot()
        logger.info(f"Decay done | deleted={deleted} remaining={self._db.get_edge_count()}")
        return deleted


# ── Smoke-test: full pipeline ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import cv2
    from core.video.stream_reader import StreamReader
    from core.tracking.person_tracker import PersonTracker
    from core.reid.embedder import PersonEmbedder
    from core.reid.identity_manager import IdentityManager
    from core.interaction.interaction_detector import InteractionDetector, annotate_interactions

    source = sys.argv[1] if len(sys.argv) > 1 else "data/cam1.mp4"
    print(f"\nStep 6 — Incident-Based Confidence Engine on: {source}")
    print("Each interaction classified → PROXIMITY / CONVERSATION / CLOSE_CONTACT / etc.")
    print("Press Q to quit.\n")

    tracker  = PersonTracker()
    embedder = PersonEmbedder()
    manager  = IdentityManager()
    detector = InteractionDetector()
    db       = GraphDB()
    engine   = ConfidenceEngine(db, decay_interval_m=1, auto_snapshot=True)
    engine.start()
    reader   = StreamReader(source=source, frame_skip=2)

    try:
        for frame_num, frame in reader.frames():
            tracked = tracker.track(frame, camera_id="cam1", frame_num=frame_num)
            persons_with_ids = []
            annotated = frame.copy()

            for person in tracked:
                vec    = embedder.embed(person.crop)
                result = manager.identify(
                    embedding=vec, track_id=person.track_id,
                    camera_id="cam1", frame_num=frame_num,
                )
                if result:
                    persons_with_ids.append((result.person_id, person.center))
                    x1, y1, x2, y2 = person.bbox
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(annotated, f"{result.person_id[-5:]}",
                                (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

            people_ids = [pid for pid, _ in persons_with_ids]
            events = detector.update(
                persons=persons_with_ids, camera_id="cam1", frame_num=frame_num
            )
            for event in events:
                engine.process_event(event, people_in_scene=people_ids)

            annotated = annotate_interactions(
                annotated, persons_with_ids,
                detector.get_active_proximities(), events,
            )
            stats = engine.get_stats()
            cv2.putText(annotated,
                        f"Graph: {stats['nodes']} people  {stats['edges']} relationships",
                        (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("URG-IS | Step 6 — Incident Graph", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        engine.stop()
        manager.save()
        reader.stop()
        cv2.destroyAllWindows()

        # Print per-person graphs
        print("\n══ PER-PERSON RELATIONSHIP GRAPHS ══════════════════════════════")
        for pid, graph in engine.get_all_person_graphs().items():
            print(f"\n▶ {pid} | {graph['total_connections']} connection(s)")
            print(f"  Cameras seen on: {graph['camera_ids']}")
            for conn in graph["connections"]:
                bar = "█" * int(conn["confidence"] * 20)
                print(f"  → {conn['person_id']:<15} "
                      f"[{conn['relationship']:<16}] "
                      f"{conn['confidence']:.3f} {bar}")
                print(f"     Incidents: {conn['incident_counts']}")
                print(f"     Meetings today: {conn['meetings_today']}  "
                      f"Avg duration: {conn['avg_duration_s']}s  "
                      f"Cameras: {conn['cameras']}")