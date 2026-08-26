"""Cost-based rate limiter — tracks cumulative LLM token cost per user in Redis.

Unlike simple request-count rate limiting, this tracks the actual dollar cost
of LLM usage per user over a rolling window.  Uses Redis INCR + EXPIRE for
atomic counter semantics without Lua scripts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as aioredis

log = logging.getLogger("default")

# Default cost limits (USD per hour / per day)
DEFAULT_HOURLY_LIMIT: float = 1.0
DEFAULT_DAILY_LIMIT: float = 5.0

_COST_KEY_PREFIX = "rag:cost:"
_HOURLY_TTL = 3600
_DAILY_TTL = 86400


class CostRateLimiter:
    """Per-user cost-based rate limiter backed by Redis.

    Usage::

        limiter = CostRateLimiter(redis_client)
        allowed, reason = await limiter.check_and_increment(user_id, cost_dollars)
        if not allowed:
            raise BusinessRuleViolation(f"Cost limit exceeded: {reason}")
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        *,
        hourly_limit: float | None = None,
        daily_limit: float | None = None,
    ) -> None:
        self._redis = redis_client
        self._hourly_limit = hourly_limit if hourly_limit is not None else DEFAULT_HOURLY_LIMIT
        self._daily_limit = daily_limit if daily_limit is not None else DEFAULT_DAILY_LIMIT

    def _hourly_key(self, user_id: int) -> str:
        return f"{_COST_KEY_PREFIX}{user_id}:hourly"

    def _daily_key(self, user_id: int) -> str:
        return f"{_COST_KEY_PREFIX}{user_id}:daily"

    async def check_and_increment(self, user_id: int, cost_dollars: float) -> tuple[bool, str]:
        """Check if user is within cost limits and increment counters.

        Returns (allowed, reason).
        If allowed=False, reason describes which limit was exceeded.
        """
        if cost_dollars <= 0:
            return True, ""

        try:
            hourly_key = self._hourly_key(user_id)
            daily_key = self._daily_key(user_id)

            # Get current costs
            hourly_raw = await self._redis.get(hourly_key)
            daily_raw = await self._redis.get(daily_key)

            hourly_cost = float(hourly_raw) if hourly_raw else 0.0
            daily_cost = float(daily_raw) if daily_raw else 0.0

            # Check limits before incrementing
            if hourly_cost + cost_dollars > self._hourly_limit:
                return False, (
                    f"Hourly cost limit exceeded: ${hourly_cost + cost_dollars:.4f} > ${self._hourly_limit}"
                )
            if daily_cost + cost_dollars > self._daily_limit:
                return False, (
                    f"Daily cost limit exceeded: ${daily_cost + cost_dollars:.4f} > ${self._daily_limit}"
                )

            # Increment atomically
            pipe = self._redis.pipeline()
            pipe.incrbyfloat(hourly_key, cost_dollars)
            pipe.expire(hourly_key, _HOURLY_TTL)
            pipe.incrbyfloat(daily_key, cost_dollars)
            pipe.expire(daily_key, _DAILY_TTL)
            await pipe.execute()

            return True, ""

        except Exception as e:
            # Fail open — allow the request if Redis is down
            log.warning("Cost rate limiter error (failing open): %s", e)
            return True, ""

    async def get_usage(self, user_id: int) -> dict[str, float]:
        """Get current cost usage for a user (for monitoring/admin endpoints)."""
        try:
            hourly_raw = await self._redis.get(self._hourly_key(user_id))
            daily_raw = await self._redis.get(self._daily_key(user_id))
            return {
                "hourly_cost": float(hourly_raw) if hourly_raw else 0.0,
                "daily_cost": float(daily_raw) if daily_raw else 0.0,
                "hourly_limit": self._hourly_limit,
                "daily_limit": self._daily_limit,
            }
        except Exception as e:
            log.warning("Cost rate limiter get_usage error: %s", e)
            return {
                "hourly_cost": 0.0,
                "daily_cost": 0.0,
                "hourly_limit": self._hourly_limit,
                "daily_limit": self._daily_limit,
            }
