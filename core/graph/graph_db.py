"""
core/graph/graph_db.py
STEP 6a — Graph Database (Incident-Based, Per-Person Graphs)

One master graph — every person is a node, every relationship is an edge.
Each edge carries full incident history, type, modifiers, and relationship label.

Per-person graph:
  db.get_person_graph("PERSON_00001")
  Returns all connections for that person — confidence, relationship type,
  incident breakdown, cameras, locations, avg duration.

Relationship labels by confidence:
  0.00–0.20 → stranger
  0.20–0.40 → acquaintance
  0.40–0.60 → associate
  0.60–0.80 → close_associate
  0.80–1.00 → significant
"""

import json
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from loguru import logger
import networkx as nx

from core.graph.incident_classifier import Incident, IncidentType
from core.graph.neo4j_store import Neo4jStore
from config.settings import ENABLE_LOUVAIN_COMMUNITIES, GRAPH_BACKEND


# ── Relationship labels ───────────────────────────────────────────────────────

def relationship_label(confidence: float) -> str:
    if confidence < 0.20: return "stranger"
    if confidence < 0.40: return "acquaintance"
    if confidence < 0.60: return "associate"
    if confidence < 0.80: return "close_associate"
    return "significant"

RELATIONSHIP_COLORS = {
    "stranger"        : "#888888",
    "acquaintance"    : "#4fc3f7",
    "associate"       : "#81c784",
    "close_associate" : "#ffb74d",
    "significant"     : "#e57373",
}


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class RelationshipEdge:
    person_id_a      : str
    person_id_b      : str
    confidence       : float
    relationship     : str
    incident_counts  : Dict[str, int]
    last_incident    : str
    meetings_today   : int
    total_duration_s : float
    cameras          : set
    locations        : list
    first_seen       : float
    last_seen        : float
    color            : str = "#888888"

    @property
    def total_meetings(self) -> int:
        return sum(self.incident_counts.values())

    @property
    def avg_duration_s(self) -> float:
        return self.total_duration_s / self.total_meetings if self.total_meetings else 0.0

    def to_dict(self) -> dict:
        return {
            "person_id_a"     : self.person_id_a,
            "person_id_b"     : self.person_id_b,
            "confidence"      : round(self.confidence, 4),
            "relationship"    : self.relationship,
            "color"           : self.color,
            "incident_counts" : self.incident_counts,
            "last_incident"   : self.last_incident,
            "meetings_today"  : self.meetings_today,
            "total_meetings"  : self.total_meetings,
            "avg_duration_s"  : round(self.avg_duration_s, 1),
            "total_duration_s": round(self.total_duration_s, 1),
            "cameras"         : list(self.cameras),
            "first_seen"      : self.first_seen,
            "last_seen"       : self.last_seen,
        }


@dataclass
class PersonNode:
    person_id         : str
    first_seen        : float
    last_seen         : float
    camera_ids        : set
    total_interactions: int = 0

    def to_dict(self) -> dict:
        return {
            "person_id"          : self.person_id,
            "first_seen"         : self.first_seen,
            "last_seen"          : self.last_seen,
            "camera_ids"         : list(self.camera_ids),
            "total_interactions" : self.total_interactions,
        }


# ── Graph Database ────────────────────────────────────────────────────────────

class GraphDB:
    """
    Master relationship graph with per-person sub-graph support.

    Usage:
        db = GraphDB()
        db.record_incident(incident, confidence=0.35)

        # Per-person view
        graph = db.get_person_graph("PERSON_00001")
        for conn in graph["connections"]:
            print(conn["person_id"], conn["relationship"], conn["confidence"])

        # Full graph for dashboard
        data = db.to_dict()
    """

    def __init__(self, snapshot_path: str = "data/snapshots/graph.json"):
        self._graph = nx.Graph()
        self._lock  = threading.RLock()
        self._snapshot_path = Path(snapshot_path)
        self._neo4j = Neo4jStore() if GRAPH_BACKEND == "neo4j" else None
        self._load_snapshot()
        logger.info(
            f"GraphDB ready | "
            f"nodes={self._graph.number_of_nodes()} | "
            f"edges={self._graph.number_of_edges()}"
        )

    # ── Core operations ───────────────────────────────────────────────────────

    def ensure_person(self, person_id: str, camera_id: str = ""):
        with self._lock:
            if not self._graph.has_node(person_id):
                self._graph.add_node(
                    person_id,
                    first_seen         = time.time(),
                    last_seen          = time.time(),
                    camera_ids         = set(),
                    total_interactions = 0,
                )
            n = self._graph.nodes[person_id]
            n["last_seen"] = time.time()
            n["total_interactions"] = n.get("total_interactions", 0) + 1
            if camera_id:
                n["camera_ids"].add(camera_id)
            if self._neo4j and self._neo4j.enabled:
                self._neo4j.upsert_person(person_id, n["last_seen"])

    def record_incident(self, incident: Incident, confidence: float) -> RelationshipEdge:
        """
        Store incident and write computed confidence to edge.
        Creates edge on first interaction, updates on repeat.
        """
        pid_a, pid_b = incident.person_id_a, incident.person_id_b
        now = time.time()

        with self._lock:
            self.ensure_person(pid_a, incident.camera_id)
            self.ensure_person(pid_b, incident.camera_id)

            if not self._graph.has_edge(pid_a, pid_b):
                self._graph.add_edge(pid_a, pid_b,
                    confidence       = confidence,
                    relationship     = relationship_label(confidence),
                    incident_counts  = defaultdict(int),
                    last_incident    = incident.incident_type.value,
                    meetings_today   = 1,
                    total_duration_s = incident.duration_s,
                    cameras          = set(),
                    locations        = [],
                    first_seen       = now,
                    last_seen        = now,
                    day_key          = self._day_key(),
                )
                logger.info(
                    f"NEW relationship | {pid_a} ↔ {pid_b} | "
                    f"{incident.incident_type.value} | "
                    f"confidence={confidence:.3f} ({relationship_label(confidence)})"
                )
            else:
                e = self._graph.edges[pid_a, pid_b]
                # Reset daily count on new day
                if e.get("day_key") != self._day_key():
                    e["meetings_today"] = 0
                    e["day_key"] = self._day_key()

                e["confidence"]       = confidence
                e["relationship"]     = relationship_label(confidence)
                e["last_incident"]    = incident.incident_type.value
                e["meetings_today"]   = e.get("meetings_today", 0) + 1
                e["total_duration_s"] = e.get("total_duration_s", 0.0) + incident.duration_s
                e["last_seen"]        = now
                logger.info(
                    f"Updated | {pid_a} ↔ {pid_b} | "
                    f"{incident.incident_type.value} | "
                    f"confidence={confidence:.3f} ({relationship_label(confidence)}) | "
                    f"meetings_today={e['meetings_today']}"
                )

            # Common updates
            e = self._graph.edges[pid_a, pid_b]
            if not isinstance(e["incident_counts"], defaultdict):
                e["incident_counts"] = defaultdict(int, e["incident_counts"])
            e["incident_counts"][incident.incident_type.value] += 1
            e["cameras"].add(incident.camera_id)
            if incident.location_px != (0, 0):
                e["locations"].append(incident.location_px)
                if len(e["locations"]) > 50:
                    e["locations"] = e["locations"][-50:]

            if self._neo4j and self._neo4j.enabled:
                self._neo4j.upsert_relationship(
                    pid_a,
                    pid_b,
                    e.get("confidence", 0.0),
                    e.get("relationship", "stranger"),
                    e.get("last_incident", ""),
                )

            return self._edge_to_obj(pid_a, pid_b)

    def apply_decay(self, multiplier: float = 0.998) -> int:
        """
        Multiplicative decay — strong relationships fade slower than weak ones.
        0.80 confidence → ~27 hrs to reach 0.40
        0.20 confidence → ~6 hrs to reach 0.10
        Deletes edges below 0.01.
        """
        with self._lock:
            to_delete = []
            for a, b, data in self._graph.edges(data=True):
                data["confidence"]   = round(data["confidence"] * multiplier, 6)
                data["relationship"] = relationship_label(data["confidence"])
                if data["confidence"] < 0.01:
                    to_delete.append((a, b))
            for a, b in to_delete:
                self._graph.remove_edge(a, b)
                logger.info(f"Edge decayed to near-zero → deleted: {a} ↔ {b}")
            return len(to_delete)

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_edge(self, pid_a: str, pid_b: str) -> Optional[RelationshipEdge]:
        with self._lock:
            if self._graph.has_edge(pid_a, pid_b):
                return self._edge_to_obj(pid_a, pid_b)
            return None

    def get_all_edges(self, max_age_seconds: Optional[float] = None) -> List[RelationshipEdge]:
        with self._lock:
            now = time.time()
            return [
                self._edge_to_obj(a, b) for a, b in self._graph.edges()
                if max_age_seconds is None or (now - self._graph.edges[a, b].get("last_seen", 0)) <= max_age_seconds
            ]

    def get_all_nodes(self, max_age_seconds: Optional[float] = None) -> List[PersonNode]:
        with self._lock:
            now = time.time()
            return [
                PersonNode(
                    person_id          = pid,
                    first_seen         = d.get("first_seen", 0),
                    last_seen          = d.get("last_seen", 0),
                    camera_ids         = d.get("camera_ids", set()),
                    total_interactions = d.get("total_interactions", 0),
                )
                for pid, d in self._graph.nodes(data=True)
                if max_age_seconds is None or (now - d.get("last_seen", 0)) <= max_age_seconds
            ]

    def get_person_graph(self, person_id: str) -> Optional[Dict]:
        """
        Personal relationship graph for one person.
        Returns all connections sorted by confidence (strongest first).
        """
        with self._lock:
            if not self._graph.has_node(person_id):
                return None

            nd = self._graph.nodes[person_id]
            connections = []

            for neighbour in self._graph.neighbors(person_id):
                edge = self._edge_to_obj(person_id, neighbour)
                connections.append({
                    "person_id"       : neighbour,
                    "confidence"      : round(edge.confidence, 4),
                    "relationship"    : edge.relationship,
                    "color"           : RELATIONSHIP_COLORS.get(edge.relationship, "#888888"),
                    "total_meetings"  : edge.total_meetings,
                    "meetings_today"  : edge.meetings_today,
                    "last_incident"   : edge.last_incident,
                    "incident_counts" : dict(edge.incident_counts),
                    "avg_duration_s"  : round(edge.avg_duration_s, 1),
                    "total_duration_s": round(edge.total_duration_s, 1),
                    "cameras"         : list(edge.cameras),
                    "first_seen"      : edge.first_seen,
                    "last_seen"       : edge.last_seen,
                })

            connections.sort(key=lambda x: x["confidence"], reverse=True)

            return {
                "person_id"          : person_id,
                "first_seen"         : nd.get("first_seen", 0),
                "last_seen"          : nd.get("last_seen", 0),
                "camera_ids"         : list(nd.get("camera_ids", set())),
                "total_connections"  : len(connections),
                "connections"        : connections,
                "significant_count"  : sum(1 for c in connections if c["relationship"] == "significant"),
                "close_count"        : sum(1 for c in connections if c["relationship"] == "close_associate"),
                "associate_count"    : sum(1 for c in connections if c["relationship"] == "associate"),
                "acquaintance_count" : sum(1 for c in connections if c["relationship"] == "acquaintance"),
            }

    def get_all_person_graphs(self) -> Dict[str, Dict]:
        return {pid: self.get_person_graph(pid) for pid in self._graph.nodes()}

    def get_node_count(self) -> int:
        with self._lock: return self._graph.number_of_nodes()

    def get_edge_count(self) -> int:
        with self._lock: return self._graph.number_of_edges()

    def get_meetings_today(self, pid_a: str, pid_b: str) -> int:
        with self._lock:
            if self._graph.has_edge(pid_a, pid_b):
                e = self._graph.edges[pid_a, pid_b]
                if e.get("day_key") == self._day_key():
                    return e.get("meetings_today", 0)
            return 0

    def to_dict(self) -> dict:
        with self._lock:
            nodes = [
                {
                    "id"                 : pid,
                    "first_seen"         : d.get("first_seen", 0),
                    "last_seen"          : d.get("last_seen", 0),
                    "camera_ids"         : list(d.get("camera_ids", set())),
                    "total_interactions" : d.get("total_interactions", 0),
                    "degree"             : self._graph.degree(pid),
                }
                for pid, d in self._graph.nodes(data=True)
            ]
            edges = [e.to_dict() for e in self.get_all_edges()]
            return {"nodes": nodes, "edges": edges, "communities": self.get_louvain_communities()}

    def save_snapshot(self):
        try:
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._snapshot_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=list)
            logger.info(
                f"Snapshot saved | "
                f"nodes={self.get_node_count()} edges={self.get_edge_count()}"
            )
        except Exception as e:
            logger.error(f"Snapshot save failed: {e}")

    def reset(self):
        with self._lock: self._graph.clear()
        logger.info("GraphDB reset.")

    def get_louvain_communities(self) -> List[dict]:
        """Automatic group discovery using Louvain community detection."""
        if not ENABLE_LOUVAIN_COMMUNITIES:
            return []
        with self._lock:
            if self._graph.number_of_nodes() == 0 or self._graph.number_of_edges() == 0:
                return []
            try:
                communities = nx.community.louvain_communities(
                    self._graph,
                    weight="confidence",
                    resolution=1.0,
                    seed=42,
                )
            except Exception as exc:
                logger.warning(f"Louvain detection unavailable: {exc}")
                return []

            out = []
            for idx, nodes in enumerate(communities, start=1):
                members = sorted(list(nodes))
                sub = self._graph.subgraph(nodes)
                out.append(
                    {
                        "community_id": idx,
                        "members": members,
                        "size": len(members),
                        "edge_count": int(sub.number_of_edges()),
                    }
                )
            return sorted(out, key=lambda c: c["size"], reverse=True)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _day_key(self) -> str:
        t = time.localtime()
        return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"

    def _edge_to_obj(self, pid_a: str, pid_b: str) -> RelationshipEdge:
        d   = self._graph.edges[pid_a, pid_b]
        rel = relationship_label(d.get("confidence", 0.0))
        return RelationshipEdge(
            person_id_a      = pid_a,
            person_id_b      = pid_b,
            confidence       = d.get("confidence", 0.0),
            relationship     = rel,
            incident_counts  = dict(d.get("incident_counts", {})),
            last_incident    = d.get("last_incident", ""),
            meetings_today   = d.get("meetings_today", 0),
            total_duration_s = d.get("total_duration_s", 0.0),
            cameras          = d.get("cameras", set()),
            locations        = d.get("locations", []),
            first_seen       = d.get("first_seen", 0),
            last_seen        = d.get("last_seen", 0),
            color            = RELATIONSHIP_COLORS.get(rel, "#888888"),
        )

    def _load_snapshot(self):
        try:
            if self._snapshot_path.exists():
                with open(self._snapshot_path) as f:
                    data = json.load(f)
                for n in data.get("nodes", []):
                    self._graph.add_node(
                        n["id"],
                        first_seen         = n.get("first_seen", 0),
                        last_seen          = n.get("last_seen", 0),
                        camera_ids         = set(n.get("camera_ids", [])),
                        total_interactions = n.get("total_interactions", 0),
                    )
                for e in data.get("edges", []):
                    self._graph.add_edge(
                        e["person_id_a"], e["person_id_b"],
                        confidence       = e.get("confidence", 0.0),
                        relationship     = e.get("relationship", "stranger"),
                        incident_counts  = defaultdict(int, e.get("incident_counts", {})),
                        last_incident    = e.get("last_incident", ""),
                        meetings_today   = e.get("meetings_today", 0),
                        total_duration_s = e.get("total_duration_s", 0.0),
                        cameras          = set(e.get("cameras", [])),
                        locations        = e.get("locations", []),
                        first_seen       = e.get("first_seen", 0),
                        last_seen        = e.get("last_seen", 0),
                        day_key          = self._day_key(),
                    )
                logger.info(f"Snapshot loaded from {self._snapshot_path}")
        except Exception as e:
            logger.warning(f"Could not load snapshot: {e} — starting fresh.")