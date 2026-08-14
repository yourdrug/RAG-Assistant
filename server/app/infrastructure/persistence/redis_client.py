"""Centralized async Redis client with explicit lifecycle.

Usage::

    from infrastructure.persistence.redis_client import redis_client

    # In lifespan startup:
    await redis_client.init()

    # Anywhere in the app:
    redis = redis_client.async_redis
    await redis.set("key", "value")

    # In lifespan shutdown:
    await redis_client.aclose()
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis
from config import settings

logger = logging.getLogger("default")


class RedisClient:
    """Async Redis client with init/close lifecycle."""

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    @property
    def async_redis(self) -> aioredis.Redis:
        """Return the async Redis connection. Raises if not initialised."""
        if self._redis is None:
            raise RuntimeError("RedisClient not initialised — call await redis_client.init() first")
        return self._redis

    async def init(self) -> None:
        """Create and verify the Redis connection."""
        if self._redis is not None:
            return

        self._redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        await self._redis.ping()
        logger.info("RedisClient: connection established to %s:%s", settings.redis_host, settings.redis_port)

    async def aclose(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("RedisClient: connection closed")


redis_client = RedisClient()
