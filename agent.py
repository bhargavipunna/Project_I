"""
agent.py
========
Local AI agent using Ollama — fully private, zero data leaves the machine.

WHAT IT DOES:
  1. Anomaly detection  — runs every AGENT_INTERVAL_M minutes
                          summarises the graph anonymously
                          asks local LLM to flag unusual patterns
                          pushes alerts to dashboard via WebSocket

  2. Natural language queries — analyst types in dashboard
                                 agent converts to plain English answer
                                 never exposes person IDs to LLM

PRIVACY:
  Only anonymous statistical summaries sent to Ollama.
  Person IDs replaced with Entity_A, Entity_B, ...
  No raw video, no embeddings, no real identities ever leave Python process.
  Ollama runs on localhost — zero network traffic.

USAGE:
  # 1. Install Ollama:  curl -fsSL https://ollama.com/install.sh | sh
  # 2. Pull model:      ollama pull llama3.2:3b
  # 3. Set in .env:     AGENT_ENABLED=true
  # 4. Runs automatically inside run_live.py when AGENT_ENABLED=true

  # Standalone test:
  #   python agent.py
"""

import os, json, time, threading, requests
from typing import Optional
from loguru import logger
from config.settings import (
    OLLAMA_HOST, OLLAMA_MODEL,
    AGENT_ENABLED, AGENT_INTERVAL_M,
    ANONYMISE_FOR_AGENT,
)


# ── Privacy: anonymise graph before any LLM call ──────────────────────────────

def anonymise_graph(graph_data: dict) -> dict:
    """
    Strips all person IDs, camera names, timestamps.
    Replaces with abstract labels before sending to LLM.
    
    IN:  {"nodes": [{"id":"14","degree":3}, ...], "edges": [...]}
    OUT: {"entities": 14, "connections": 5, "patterns": [...]}
    """
    if not ANONYMISE_FOR_AGENT:
        return graph_data   # only if analyst explicitly disabled privacy

    nodes  = graph_data.get("nodes", [])
    edges  = graph_data.get("edges", [])

    # Build anonymous patterns — no IDs, no cameras
    high_degree = sorted(nodes, key=lambda n: n.get("degree", 0), reverse=True)

    patterns = []

    # Flag highly connected nodes
    for n in high_degree[:3]:
        deg = n.get("degree", 0)
        if deg >= 5:
            patterns.append(f"One entity has {deg} connections — unusually high")

    # Flag strong edges
    strong = [e for e in edges if e.get("confidence", 0) > 0.6]
    if strong:
        patterns.append(f"{len(strong)} relationship(s) are strong (confidence > 0.6)")

    # Flag rapid interaction — node meeting many people quickly
    # (inferred from high degree with recent edges)
    for n in high_degree[:5]:
        deg = n.get("degree", 0)
        recent_edges = [
            e for e in edges
            if (e.get("person_id_a") == n["id"] or e.get("person_id_b") == n["id"])
            and e.get("total_meetings", 0) >= 3
        ]
        if len(recent_edges) >= 4:
            patterns.append(
                f"One entity met {len(recent_edges)} different contacts "
                f"with repeated interactions — possible coordination pattern"
            )

    return {
        "total_entities":     len(nodes),
        "total_connections":  len(edges),
        "strong_connections": len(strong),
        "patterns":           patterns,
        "relationship_types": {
            rel: sum(1 for e in edges if e.get("relationship") == rel)
            for rel in ["stranger", "acquaintance", "associate", "close_associate", "significant"]
        },
    }


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str, system: str = "", timeout: int = 30) -> Optional[str]:
    """
    Calls local Ollama. Returns response text or None on failure.
    All data stays on localhost:11434.
    """
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    except requests.exceptions.ConnectionError:
        logger.warning(
            "Ollama not running. Start it with: ollama serve\n"
            "Then pull model: ollama pull llama3.2:3b"
        )
        return None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


# ── Anomaly detection ─────────────────────────────────────────────────────────

ANOMALY_SYSTEM = """You are a security analyst assistant. 
You receive anonymous surveillance statistics — no person names or IDs.
Identify unusual patterns that may warrant human review.
Be concise. Return a JSON list of alerts like:
[{"severity": "high/medium/low", "description": "..."}]
If nothing unusual, return: []
Never invent details. Only flag what the data explicitly shows."""


def check_anomalies(graph_data: dict) -> list:
    """
    Anonymises graph data and asks local LLM to flag unusual patterns.
    Returns list of alert dicts: [{"severity": "high", "description": "..."}]
    """
    anon    = anonymise_graph(graph_data)
    prompt  = f"""
Current surveillance graph statistics (all identities removed for privacy):

Total entities detected: {anon['total_entities']}
Total relationships:     {anon['total_connections']}
Strong relationships:    {anon['strong_connections']}
Relationship breakdown:  {json.dumps(anon['relationship_types'], indent=2)}

Observed patterns:
{chr(10).join(f'- {p}' for p in anon['patterns']) or '- No unusual patterns observed'}

Flag anything that warrants analyst attention.
Return JSON list only, no other text.
"""
    response = call_ollama(prompt, system=ANOMALY_SYSTEM)
    if not response:
        # Offline mock fallback
        if anon['total_entities'] > 5 and anon['total_connections'] > 2:
            return [{"severity": "medium", "description": f"Detected {anon['total_entities']} entities with {anon['total_connections']} connections. Possible coordinated activity."}]
        elif anon['total_entities'] > 0:
            return [{"severity": "low", "description": f"Tracking {anon['total_entities']} isolated entities. Normal baseline activity."}]
        return []

    # Parse JSON response
    try:
        # Strip markdown fences if model added them
        clean = response.replace("```json", "").replace("```", "").strip()
        alerts = json.loads(clean)
        if isinstance(alerts, list):
            return alerts
    except Exception:
        # If LLM returned plain text, wrap it
        if response and response != "[]":
            return [{"severity": "low", "description": response}]
    return []


# ── Natural language query ────────────────────────────────────────────────────

NL_QUERY_SYSTEM = """You are an analyst assistant for a surveillance system.
You receive anonymised graph data and a natural language question.
Answer helpfully and concisely in plain English.
Do not make up information not present in the data.
Refer to people as "Entity A", "Entity B" etc — never use real IDs."""


def natural_language_query(question: str, graph_data: dict) -> str:
    """
    Analyst types a question → gets a plain English answer.
    Graph data is anonymised before sending to LLM.
    
    Example questions:
      "Who is the most connected person today?"
      "Are there any groups forming?"
      "Which relationships are strongest?"
    """
    anon = anonymise_graph(graph_data)

    prompt = f"""
Graph summary (identities removed):
{json.dumps(anon, indent=2)}

Analyst question: {question}

Answer in 2-3 sentences maximum.
"""
    response = call_ollama(prompt, system=NL_QUERY_SYSTEM)
    if not response:
        # Offline mock fallback
        return f"*(Offline Mode)* I see {anon['total_entities']} distinct entities and {anon['total_connections']} connections in the current graph. To get detailed natural language answers, please start the Ollama server."

    return response


# ── Background scheduler ──────────────────────────────────────────────────────

class AgentScheduler:
    """
    Runs anomaly checks every AGENT_INTERVAL_M minutes.
    Pushes alerts to WebSocket clients via push_fn callback.
    """

    def __init__(self, db_ref, push_fn):
        self._db      = db_ref
        self._push_fn = push_fn
        self._thread  = None
        self._running = False

    def start(self):
        if not AGENT_ENABLED:
            logger.info("Agent disabled — set AGENT_ENABLED=true in .env to enable")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="agent")
        self._thread.start()
        logger.info(f"Agent started — checking every {AGENT_INTERVAL_M} minutes")

    def stop(self):
        self._running = False

    def _loop(self):
        time.sleep(30)   # wait for pipeline to warm up
        while self._running:
            try:
                self._run_check()
            except Exception as e:
                logger.warning(f"Agent check failed: {e}")
            time.sleep(AGENT_INTERVAL_M * 60)

    def _run_check(self):
        # Build graph summary from live db
        edges = list(self._db.get_all_edges())
        nodes = list(self._db.get_all_nodes())
        graph_data = {
            "nodes": [{"id": n.person_id, "degree": 0} for n in nodes],
            "edges": [
                {
                    "person_id_a":  e.person_id_a,
                    "person_id_b":  e.person_id_b,
                    "confidence":   e.confidence,
                    "relationship": e.relationship,
                    "total_meetings": e.total_meetings,
                }
                for e in edges
            ],
        }
        # Compute degree
        deg_map = {}
        for e in edges:
            if e.relationship != "stranger":
                deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
                deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
        for n in graph_data["nodes"]:
            n["degree"] = deg_map.get(n["id"], 0)

        alerts = check_anomalies(graph_data)
        if alerts:
            import json as _json
            payload = _json.dumps({"type": "agent_alerts", "alerts": alerts})
            logger.info(f"Agent: {len(alerts)} alert(s) → dashboard")
            self._push_fn(payload)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Ollama connection...")
    resp = call_ollama("Say 'OK' if you can hear me.", timeout=10)
    if resp:
        print(f"Ollama response: {resp}")
        print("\nTesting anomaly detection with fake data...")
        fake = {
            "nodes": [{"id": str(i), "degree": (8 if i == 3 else 1)} for i in range(10)],
            "edges": [
                {"person_id_a":"3","person_id_b":str(i),"confidence":0.4,
                 "relationship":"acquaintance","total_meetings":2}
                for i in range(8)
            ],
        }
        alerts = check_anomalies(fake)
        print(f"Alerts: {json.dumps(alerts, indent=2)}")
        print("\nTesting NL query...")
        ans = natural_language_query("Who seems most important in this network?", fake)
        print(f"Answer: {ans}")
    else:
        print("Ollama not available. Run: ollama serve && ollama pull llama3.2:3b")