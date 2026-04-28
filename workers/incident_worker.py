"""
Redis Stream worker for async incident handling.

Run:
  python workers/incident_worker.py --name worker-1
"""

from __future__ import annotations

import argparse

from loguru import logger

from core.streaming.redis_streams import RedisIncidentStream


def handle_incident(payload: dict):
    # Placeholder: attach heavy enrichment / alerts / downstream sinks here.
    logger.info(
        "Incident worker processed "
        f"{payload.get('person_id_a')} ↔ {payload.get('person_id_b')} "
        f"type={payload.get('incident_type')} conf={payload.get('confidence')}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="worker-1")
    args = parser.parse_args()

    stream = RedisIncidentStream()
    if not stream.enabled:
        logger.error("Redis is not available. Worker exiting.")
        return

    logger.info(f"Starting incident worker {args.name}")
    stream.consume_forever(consumer_name=args.name, handler=handle_incident)


if __name__ == "__main__":
    main()
