"""Redis-backed rate limiter implementing ``RateLimiterPort``.

One pyrate-limiter ``Limiter`` shared by all policies — the bucket factory routes
each item name to its own per-principal RedisBucket with policy-specific rates.
Fail-open: backend errors are logged/counted and the request is allowed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyrate_limiter import Limiter
from redis.exceptions import RedisError

from application.ports.rate_limit import (
    RateLimitDecision,
    RateLimitPolicy,
    RateLimitPolicyName,
)
from infrastructure.metrics.metrics import (
    RATE_LIMIT_BACKEND_ERRORS_TOTAL,
    RATE_LIMIT_EXCEEDED_TOTAL,
)
from infrastructure.rate_limit.bucket_factory import RedisBucketFactory

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger("default")

_BACKEND_ERRORS = (RedisError, OSError, TimeoutError)


class PyrateRateLimiter:
    """Distributed rate limiter backed by per-principal Redis sorted sets."""

    def __init__(
            self,
            redis: Redis,
            policies: dict[RateLimitPolicyName, RateLimitPolicy],
            *,
            key_prefix: str,
            max_buckets: int,
            fail_open: bool = True,
    ) -> None:
        if not policies:
            raise ValueError("policies must not be empty")
        self._policies = policies
        self._fail_open = fail_open
        self._factory = RedisBucketFactory(
            redis, policies, key_prefix=key_prefix, max_buckets=max_buckets
        )
        self._limiter = Limiter(self._factory)
        self._tightest_limit = {
            name: min(policy.rates, key=lambda spec: spec.interval_ms).limit
            for name, policy in policies.items()
        }

    async def check(self, policy_name: RateLimitPolicyName, principal: str) -> RateLimitDecision:
        policy = self._policies.get(policy_name)
        if policy is None:
            raise ValueError(f"Unknown rate limit policy: {policy_name!r}")

        item_name = f"{policy_name.value}:{principal}"
        try:
            allowed = await self._limiter.try_acquire_async(item_name, blocking=False)
        except _BACKEND_ERRORS as exc:
            RATE_LIMIT_BACKEND_ERRORS_TOTAL.labels(policy=policy_name.value).inc()
            logger.warning(
                "Rate limit backend error (policy=%s): [%s] %s",
                policy_name.value, type(exc).__name__, exc,
            )
            if self._fail_open:
                return RateLimitDecision(allowed=True, degraded=True)
            return RateLimitDecision(
                allowed=False,
                retry_after_sec=policy.retry_after_sec,
                limit=self._tightest_limit[policy_name],
                degraded=True,
            )

        if not allowed:
            RATE_LIMIT_EXCEEDED_TOTAL.labels(policy=policy_name.value).inc()
            limit, retry_after_sec = await self._resolve_violation(item_name, policy, policy_name)
            return RateLimitDecision(
                allowed=False,
                retry_after_sec=retry_after_sec,
                limit=limit,
            )

        await self._refresh_ttl(item_name)
        return RateLimitDecision(allowed=True, limit=self._tightest_limit[policy_name])

    async def aclose(self) -> None:
        """Close the limiter and all registered buckets.

        ``Limiter.close()`` is synchronous but only clears dicts and sets a
        threading.Event — cheap and non-blocking, safe to call from the event loop.
        """
        self._limiter.close()

    async def _refresh_ttl(self, item_name: str) -> None:
        try:
            await self._factory.refresh_key_ttl(item_name)
        except _BACKEND_ERRORS as exc:  # best-effort TTL refresh must not fail the request
            logger.debug("Rate limit TTL refresh failed for %s: %s", item_name, exc)

    async def _resolve_violation(
            self,
            item_name: str,
            policy: RateLimitPolicy,
            policy_name: RateLimitPolicyName,
    ) -> tuple[int, int]:
        """Best-effort: identify which window was actually violated.

        Falls back to the pre-computed tightest limit / configured retry_after
        if the bucket can't be inspected for any reason — a wrong "limit" in
        the response is cosmetic, so this must never raise or block the reply.
        """
        fallback = (self._tightest_limit[policy_name], policy.retry_after_sec)
        try:
            bucket = self._factory.buckets.get(item_name)
            if bucket is None:
                return fallback

            failing_rate = getattr(bucket, "failing_rate", None)
            if failing_rate is None:
                return fallback

            retry_after_sec = max(1, failing_rate.interval // 1000)
            return failing_rate.limit, retry_after_sec
        except Exception:  # never let diagnostics break the deny path
            logger.debug("Could not resolve violated rate for %s", item_name, exc_info=True)
            return fallback
