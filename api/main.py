"""
api/main.py
STEP 7 — FastAPI Backend + WebSocket Server

Provides:
  REST endpoints  → query graph data, person profiles, stats
  WebSocket       → streams live graph updates to dashboard in real time

Endpoints:
  GET  /                        → health check
  GET  /api/stats               → system stats (nodes, edges, incidents)
  GET  /api/graph               → full graph (all nodes + edges)
  GET  /api/person/{person_id}  → personal relationship graph
  GET  /api/persons             → list of all known people
  GET  /api/edge/{a}/{b}        → relationship between two people
  GET  /api/incidents/{a}/{b}   → confidence history for a pair
  POST /api/decay               → manually trigger decay
  WS   /ws/graph                → live graph stream (pushes on every change)
  WS   /ws/person/{person_id}   → live stream for one person's graph
"""

import asyncio
import json
import time
from typing import Dict, Set, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.graph.graph_db import GraphDB
from core.graph.confidence_engine import ConfidenceEngine


# ── App state shared between pipeline and API ─────────────────────────────────

class AppState:
    """
    Singleton state container.
    Pipeline writes here → API reads from here → WebSocket pushes to dashboard.
    """
    def __init__(self):
        self.db     : Optional[GraphDB]         = None
        self.engine : Optional[ConfidenceEngine] = None
        self.pipeline_running : bool = False
        self.last_update      : float = 0.0
        self.frame_count      : int   = 0
        self.incident_count   : int   = 0

app_state = AppState()


# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    """
    Tracks all active WebSocket connections.
    Supports broadcast (all clients) and targeted (per person) channels.
    """

    def __init__(self):
        # All clients subscribed to full graph updates
        self._graph_clients   : Set[WebSocket] = set()
        # Clients subscribed to a specific person's graph
        self._person_clients  : Dict[str, Set[WebSocket]] = {}

    async def connect_graph(self, ws: WebSocket):
        await ws.accept()
        self._graph_clients.add(ws)
        logger.info(f"WS client connected (graph) | total={len(self._graph_clients)}")

    async def connect_person(self, ws: WebSocket, person_id: str):
        await ws.accept()
        if person_id not in self._person_clients:
            self._person_clients[person_id] = set()
        self._person_clients[person_id].add(ws)
        logger.info(f"WS client connected (person={person_id})")

    def disconnect_graph(self, ws: WebSocket):
        self._graph_clients.discard(ws)

    def disconnect_person(self, ws: WebSocket, person_id: str):
        if person_id in self._person_clients:
            self._person_clients[person_id].discard(ws)

    async def broadcast_graph(self, data: dict):
        """Send full graph update to all connected graph clients."""
        if not self._graph_clients:
            return
        message = json.dumps({"type": "graph_update", "data": data, "ts": time.time()})
        dead = set()
        for ws in self._graph_clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self._graph_clients -= dead

    async def broadcast_person(self, person_id: str, data: dict):
        """Send personal graph update to clients watching this person."""
        clients = self._person_clients.get(person_id, set())
        if not clients:
            return
        message = json.dumps({"type": "person_update", "person_id": person_id,
                               "data": data, "ts": time.time()})
        dead = set()
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self._person_clients[person_id] -= dead

    @property
    def graph_client_count(self) -> int:
        return len(self._graph_clients)


ws_manager = ConnectionManager()


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise graph DB and confidence engine on startup."""
    logger.info("URG-IS API starting up...")

    app_state.db     = GraphDB()
    app_state.engine = ConfidenceEngine(
        app_state.db,
        decay_interval_m = 10,
        auto_snapshot    = True,
    )
    app_state.engine.start()

    # Start background task that pushes graph updates to WebSocket clients
    asyncio.create_task(_graph_push_loop())

    logger.success("URG-IS API ready.")
    yield

    # Shutdown
    logger.info("URG-IS API shutting down...")
    if app_state.engine:
        app_state.engine.stop()


app = FastAPI(
    title       = "URG-IS API",
    description = "Universal Relationship Graph Intelligence System",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # dashboard can be on any port
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {
        "status"  : "running",
        "service" : "URG-IS",
        "version" : "1.0.0",
        "pipeline": app_state.pipeline_running,
        "uptime"  : time.time(),
    }


@app.get("/api/stats")
async def get_stats():
    """System-wide statistics."""
    _check_ready()
    stats = app_state.engine.get_stats()
    stats["pipeline_running"] = app_state.pipeline_running
    stats["frame_count"]      = app_state.frame_count
    stats["last_update"]      = app_state.last_update
    stats["ws_clients"]       = ws_manager.graph_client_count
    return stats


@app.get("/api/graph")
async def get_full_graph():
    """
    Full relationship graph — all nodes and edges.
    Used by the main dashboard graph visualisation.
    """
    _check_ready()
    data = app_state.db.to_dict()

    # Add relationship colors to edges for direct use in dashboard
    from core.graph.graph_db import RELATIONSHIP_COLORS
    for edge in data["edges"]:
        edge["color"] = RELATIONSHIP_COLORS.get(edge.get("relationship", "stranger"), "#888888")

    return data


@app.get("/api/persons")
async def get_all_persons():
    """List of all known people with basic info."""
    _check_ready()
    nodes = app_state.db.get_all_nodes()
    return {
        "count"  : len(nodes),
        "persons": [n.to_dict() for n in nodes],
    }


@app.get("/api/person/{person_id}")
async def get_person(person_id: str):
    """
    Personal relationship graph for one person.
    Returns all their connections with confidence scores,
    relationship types, incident breakdowns.
    """
    _check_ready()
    graph = app_state.db.get_person_graph(person_id)
    if not graph:
        raise HTTPException(status_code=404, detail=f"{person_id} not found")
    return graph


@app.get("/api/edge/{person_id_a}/{person_id_b}")
async def get_edge(person_id_a: str, person_id_b: str):
    """Relationship between two specific people."""
    _check_ready()
    edge = app_state.db.get_edge(person_id_a, person_id_b)
    if not edge:
        raise HTTPException(
            status_code=404,
            detail=f"No relationship found between {person_id_a} and {person_id_b}"
        )
    return edge.to_dict()


@app.get("/api/incidents/{person_id_a}/{person_id_b}")
async def get_incident_history(person_id_a: str, person_id_b: str):
    """
    Full confidence history for a pair.
    Returns list of (timestamp, confidence, incident_type).
    """
    _check_ready()
    log = app_state.engine.get_confidence_log(person_id_a, person_id_b)
    return {
        "person_id_a" : person_id_a,
        "person_id_b" : person_id_b,
        "history"     : [
            {"timestamp": ts, "confidence": conf, "incident_type": inc}
            for ts, conf, inc in log
        ],
    }


@app.post("/api/decay")
async def trigger_decay():
    """Manually trigger one decay run."""
    _check_ready()
    deleted = app_state.engine.force_decay()
    return {
        "deleted_edges" : deleted,
        "remaining"     : app_state.db.get_edge_count(),
    }


# ── WebSocket endpoints ───────────────────────────────────────────────────────

@app.websocket("/ws/graph")
async def ws_graph(websocket: WebSocket):
    """
    Live full graph stream.
    Sends graph update whenever the pipeline processes new incidents.
    Dashboard subscribes here for live visualisation.
    """
    await ws_manager.connect_graph(websocket)
    try:
        # Send current graph immediately on connect
        if app_state.db:
            data = app_state.db.to_dict()
            await websocket.send_text(json.dumps({
                "type": "graph_update",
                "data": data,
                "ts"  : time.time(),
            }))

        # Keep connection alive — updates sent via broadcast_graph()
        while True:
            # Ping/pong to detect disconnects
            await asyncio.sleep(5)
            await websocket.send_text(json.dumps({"type": "ping", "ts": time.time()}))

    except WebSocketDisconnect:
        ws_manager.disconnect_graph(websocket)
        logger.info("WS graph client disconnected.")


@app.websocket("/ws/person/{person_id}")
async def ws_person(websocket: WebSocket, person_id: str):
    """
    Live personal graph stream for one person.
    Dashboard subscribes here when user clicks on a person node.
    """
    await ws_manager.connect_person(websocket, person_id)
    try:
        # Send current personal graph immediately
        if app_state.db:
            graph = app_state.db.get_person_graph(person_id)
            if graph:
                await websocket.send_text(json.dumps({
                    "type"      : "person_update",
                    "person_id" : person_id,
                    "data"      : graph,
                    "ts"        : time.time(),
                }))

        while True:
            await asyncio.sleep(5)
            await websocket.send_text(json.dumps({"type": "ping", "ts": time.time()}))

    except WebSocketDisconnect:
        ws_manager.disconnect_person(websocket, person_id)
        logger.info(f"WS person client disconnected: {person_id}")


# ── Pipeline integration ──────────────────────────────────────────────────────

async def notify_graph_updated(persons_updated: list = None):
    """
    Called by pipeline after every batch of incidents.
    Pushes updated graph to all WebSocket clients.
    Also pushes personal graph updates for affected people.
    """
    if not app_state.db:
        return

    app_state.last_update = time.time()

    # Broadcast full graph
    data = app_state.db.to_dict()
    await ws_manager.broadcast_graph(data)

    # Broadcast personal graphs for affected people
    if persons_updated:
        for pid in persons_updated:
            person_graph = app_state.db.get_person_graph(pid)
            if person_graph:
                await ws_manager.broadcast_person(pid, person_graph)


# ── Background graph push loop ────────────────────────────────────────────────

async def _graph_push_loop():
    """
    Pushes graph state to all WebSocket clients every 2 seconds.
    Acts as a heartbeat so the dashboard always shows current state
    even if no new incidents are happening.
    """
    while True:
        await asyncio.sleep(2)
        if app_state.db and ws_manager.graph_client_count > 0:
            try:
                data = app_state.db.to_dict()
                await ws_manager.broadcast_graph(data)
            except Exception as e:
                logger.debug(f"Graph push error: {e}")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_ready():
    if not app_state.db or not app_state.engine:
        raise HTTPException(status_code=503, detail="System not initialised")


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT

    print(f"\nURG-IS API starting on http://{API_HOST}:{API_PORT}")
    print("Endpoints:")
    print(f"  GET  http://localhost:{API_PORT}/api/graph")
    print(f"  GET  http://localhost:{API_PORT}/api/persons")
    print(f"  GET  http://localhost:{API_PORT}/api/person/PERSON_00001")
    print(f"  GET  http://localhost:{API_PORT}/api/stats")
    print(f"  WS   ws://localhost:{API_PORT}/ws/graph")
    print(f"  WS   ws://localhost:{API_PORT}/ws/person/PERSON_00001")
    print("\nOpen http://localhost:8000/docs for interactive API docs\n")

    uvicorn.run(
        "api.main:app",
        host    = API_HOST,
        port    = API_PORT,
        reload  = False,
        workers = 1,
    )