"""
Redis Streams publisher/consumer helpers for fault-tolerant workers.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Optional

from loguru import logger

from config.settings import (
    REDIS_CONSUMER_GROUP,
    REDIS_STREAM_INCIDENTS,
    REDIS_STREAM_MAXLEN,
    REDIS_URL,
)


class RedisIncidentStream:
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        stream_name: str = REDIS_STREAM_INCIDENTS,
        maxlen: int = REDIS_STREAM_MAXLEN,
        group_name: str = REDIS_CONSUMER_GROUP,
    ):
        self.stream_name = stream_name
        self.maxlen = maxlen
        self.group_name = group_name
        self._client = None
        try:
            import redis

            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            logger.info(f"RedisIncidentStream connected | stream={stream_name}")
        except Exception as exc:
            logger.warning(f"Redis unavailable, stream disabled: {exc}")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def publish_incident(self, payload: Dict) -> Optional[str]:
        if not self._client:
            return None
        message = {"payload": json.dumps(payload)}
        return self._client.xadd(self.stream_name, message, maxlen=self.maxlen, approximate=True)

    def ensure_group(self) -> bool:
        if not self._client:
            return False
        try:
            self._client.xgroup_create(self.stream_name, self.group_name, id="0", mkstream=True)
        except Exception:
            pass
        return True

    def consume_forever(self, consumer_name: str, handler: Callable[[Dict], None], block_ms: int = 5000):
        if not self._client:
            logger.error("Redis consumer cannot start: no client")
            return
        self.ensure_group()
        while True:
            records = self._client.xreadgroup(
                groupname=self.group_name,
                consumername=consumer_name,
                streams={self.stream_name: ">"},
                count=20,
                block=block_ms,
            )
            if not records:
                continue
            for _, entries in records:
                for message_id, fields in entries:
                    try:
                        payload = json.loads(fields.get("payload", "{}"))
                        handler(payload)
                        self._client.xack(self.stream_name, self.group_name, message_id)
                    except Exception as exc:
                        logger.error(f"Worker handler failed for {message_id}: {exc}")
