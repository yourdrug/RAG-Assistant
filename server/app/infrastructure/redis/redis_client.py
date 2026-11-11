"""Centralized async Redis client with explicit lifecycle.

Usage::

    from infrastructure.redis.redis_client import redis_client

    # In lifespan startup:
    await redis_client.init()

    # Anywhere in the app:
    redis = redis_client.async_redis
    await redis.set("key", "value")

    # In lifespan shutdown:
    await redis_client.aclose()

Lifecycle:
  - Created at import time (module-level ``redis_client`` singleton)
  - ``init()`` called in FastAPI ``lifespan()``
  - ``aclose()`` called in FastAPI ``lifespan()`` shutdown
  - Application-scoped: one connection pool per process
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from config import RedisSection, process_section


logger = logging.getLogger("default")


class RedisClient:
    """Async Redis client with init/close lifecycle."""

    def __init__(self, redis_settings: RedisSection) -> None:
        self._settings = redis_settings
        self._redis: aioredis.Redis | None = None

    @property
    def async_redis(self) -> aioredis.Redis:
        """Return the async Redis connection. Raises if not initialised."""
        if self._redis is None:
            msg = "RedisClient not initialised — call await redis_client.init() first"
            raise RuntimeError(msg)

        return self._redis

    async def init(self) -> None:
        """Create and verify the Redis connection."""
        redis_settings = self._settings

        if self._redis is not None:
            return

        client = aioredis.Redis.from_url(
            redis_settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=redis_settings.redis_socket_timeout,
            max_connections=redis_settings.redis_max_connections,
            health_check_interval=30,
        )
        try:
            await client.ping()
        except BaseException:
            await client.aclose()
            raise
        self._redis = client
        logger.info(
            "RedisClient: connection established to %s:%s",
            redis_settings.redis_host,
            redis_settings.redis_port,
        )

    async def aclose(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("RedisClient: connection closed")


# Process-wide singleton: the section is the live settings object, so hot-reload still applies.
redis_client = RedisClient(process_section(RedisSection))
