"""Redis-backed idempotency key store for write operations.

Prevents duplicate execution of write requests (documents upload, chat, benchmark)
when clients retry due to network timeouts.

Usage::

    store = IdempotencyStore(redis)
    cached = await store.get(key="uuid-1234", user_id=42)
    if cached:
        return cached  # return cached response
    # ... execute operation ...
    await store.store(key="uuid-1234", user_id=42, result={"id": 1}, ttl=3600)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from redis.exceptions import RedisError

logger = logging.getLogger("default")

IDEMPOTENCY_PREFIX = "rag:idempotency:v1:"
DEFAULT_TTL = 3600  # 1 hour


class IdempotencyStore:
    """Redis-backed store for idempotency keys."""

    def __init__(self, redis, ttl: int = DEFAULT_TTL) -> None:
        self._redis = redis
        self._ttl = ttl

    def _redis_key(self, key: str, user_id: int) -> str:
        return f"{IDEMPOTENCY_PREFIX}{user_id}:{key}"

    async def get(self, key: str, user_id: int) -> dict | None:
        """Find cached result for idempotency key. None = no duplicate."""
        try:
            raw = await self._redis.get(self._redis_key(key, user_id))
            if raw is None:
                return None
            return json.loads(raw)
        except (RedisError, json.JSONDecodeError) as exc:
            logger.warning("IdempotencyStore.get failed: %s", exc)
            return None

    async def store(self, key: str, user_id: int, result: dict[str, Any]) -> None:
        """Save result with TTL. Silently fails on Redis errors."""
        try:
            await self._redis.setex(
                self._redis_key(key, user_id),
                self._ttl,
                json.dumps(result, default=str),
            )
        except RedisError as exc:
            logger.warning("IdempotencyStore.store failed: %s", exc)
