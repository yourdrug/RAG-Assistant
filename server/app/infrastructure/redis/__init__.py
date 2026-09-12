"""Redis client — centralized async Redis connection with lifecycle management."""

from infrastructure.redis.redis_client import RedisClient, redis_client  # noqa: F401

__all__ = ["RedisClient", "redis_client"]
