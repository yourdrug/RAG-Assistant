"""Static API key provider -- generation, hashing, and Redis-backed cache.

Static API keys are an authentication method for external (kind='client')
users only. Unlike JWT, the key is verified against the database (via a
short-TTL cache) and can be revoked instantly without cryptography -- similar
to Stripe/OpenAI key schemes.

Cache is backed by Redis with TTL-based expiry (30 s). Redis is a mandatory
component.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets

from infrastructure.persistence.redis_client import redis_client

logger = logging.getLogger("default")

_KEY_PREFIX = "rg_sys_"
_CACHE_TTL_SECONDS = 30.0  # намеренно короткий TTL: отзыв ключа должен применяться быстро
_REDIS_CACHE_PREFIX = "api_key:"
_REDIS_ID_INDEX_PREFIX = "api_key_id:"
_REDIS_REVOKED_CHANNEL = "api_key_revoked"

MISS = object()


class ApiKeyProvider:
    """Кэширует sha256(ключ) -> данные пользователя на несколько секунд.

    Cache is stored in Redis with TTL (SETEX/GET).
    """

    MISS = MISS

    @staticmethod
    def generate_key() -> str:
        """Сгенерировать новый ключ в открытом виде. Показывается пользователю ОДИН РАЗ."""
        return f"{_KEY_PREFIX}{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Детерминированный sha256-хеш для хранения/поиска."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    def key_prefix_for_display(raw_key: str) -> str:
        return raw_key[: len(_KEY_PREFIX) + 6]

    # ------------------------------------------------------------------
    # Public cache API
    # ------------------------------------------------------------------

    async def get_cached(self, key_hash: str):
        """Return cached value or ``MISS``."""
        try:
            raw = await redis_client.async_redis.get(f"{_REDIS_CACHE_PREFIX}{key_hash}")
            if raw is None:
                return MISS
            return json.loads(raw)
        except Exception:
            logger.warning("Redis GET failed for api_key cache, treating as MISS")
            return MISS

    async def set_cached(self, key_hash: str, value: dict | None) -> None:
        """Store value in cache with TTL."""
        try:
            if value is None:
                await redis_client.async_redis.setex(
                    f"{_REDIS_CACHE_PREFIX}{key_hash}",
                    int(_CACHE_TTL_SECONDS),
                    json.dumps(None),
                )
            else:
                await redis_client.async_redis.setex(
                    f"{_REDIS_CACHE_PREFIX}{key_hash}",
                    int(_CACHE_TTL_SECONDS),
                    json.dumps(value),
                )
                # Maintain id -> key_hash index for invalidation
                await redis_client.async_redis.setex(
                    f"{_REDIS_ID_INDEX_PREFIX}{value['api_key_id']}",
                    int(_CACHE_TTL_SECONDS),
                    key_hash,
                )
        except Exception:
            logger.warning("Redis SETEX failed for api_key cache")

    async def invalidate_by_id(self, api_key_id: int) -> None:
        """Instantly invalidate cache for a specific API key.

        Deletes the cache entry AND publishes to ``api_key_revoked`` channel
        so all instances drop their local references immediately.
        """
        try:
            # Look up key_hash from id index
            key_hash = await redis_client.async_redis.get(f"{_REDIS_ID_INDEX_PREFIX}{api_key_id}")
            if key_hash is not None:
                await redis_client.async_redis.delete(f"{_REDIS_CACHE_PREFIX}{key_hash}")
                await redis_client.async_redis.delete(f"{_REDIS_ID_INDEX_PREFIX}{api_key_id}")

            # Publish revocation event to all instances
            await redis_client.async_redis.publish(
                _REDIS_REVOKED_CHANNEL,
                json.dumps({"api_key_id": api_key_id}),
            )
            logger.info(
                "ApiKeyProvider: revoked key id=%d, published to %s", api_key_id, _REDIS_REVOKED_CHANNEL
            )
        except Exception:
            logger.warning("Redis invalidation failed for api_key id=%d", api_key_id)


api_key_provider = ApiKeyProvider()
