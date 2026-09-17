"""Tests for the Redis-backed pyrate-limiter adapter (fakeredis + Lua)."""

from __future__ import annotations

import sys
from pathlib import Path

import fakeredis.aioredis
import pytest
from redis.exceptions import RedisError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from application.ports.rate_limit import (  # noqa: E402
    RateLimitPolicy,
    RateLimitPolicyName,
    RateSpec,
)
from infrastructure.rate_limit.bucket_factory import RedisBucketFactory  # noqa: E402
from infrastructure.rate_limit.limiter import PyrateRateLimiter  # noqa: E402
from pyrate_limiter import RateItem  # noqa: E402

_KEY_PREFIX = "test:ratelimit:"


def _policies() -> dict[RateLimitPolicyName, RateLimitPolicy]:
    return {
        RateLimitPolicyName.LOGIN: RateLimitPolicy(
            name=RateLimitPolicyName.LOGIN,
            rates=(RateSpec(2, 60_000),),
            retry_after_sec=60,
        ),
        RateLimitPolicyName.SEARCH: RateLimitPolicy(
            name=RateLimitPolicyName.SEARCH,
            rates=(RateSpec(3, 60_000),),
            retry_after_sec=60,
        ),
    }


class _BrokenRedis:
    """Redis stand-in that always fails with a backend error."""

    async def script_load(self, *args, **kwargs):
        raise RedisError("redis is down")

    async def evalsha(self, *args, **kwargs):  # pragma: no cover - script_load fails first
        raise RedisError("redis is down")


class TestResolveViolation:
    """Test that _resolve_violation returns the actual violated rate, not just the tightest."""

    @pytest.mark.asyncio
    async def test_single_window_returns_correct_limit(self):
        limiter = PyrateRateLimiter(
            fakeredis.aioredis.FakeRedis(decode_responses=True),
            _policies(),
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
        )

        # Exhaust LOGIN limit (2/min)
        await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")
        await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")
        denied = await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")

        assert denied.allowed is False
        assert denied.limit == 2  # LOGIN window limit
        assert denied.retry_after_sec == 60  # LOGIN window interval

        await limiter.aclose()

    @pytest.mark.asyncio
    async def test_multi_window_violation_returns_correct_rate(self):
        """When the minute window is exceeded, _resolve_violation returns the
        minute rate (limit=10, retry=60), NOT the daily rate (100/day)."""
        chat_policy = {
            RateLimitPolicyName.CHAT: RateLimitPolicy(
                name=RateLimitPolicyName.CHAT,
                rates=(
                    RateSpec(10, 60_000),       # 10/мин
                    RateSpec(100, 86_400_000),   # 100/день
                ),
                retry_after_sec=60,
            )
        }
        limiter = PyrateRateLimiter(
            fakeredis.aioredis.FakeRedis(decode_responses=True),
            chat_policy,
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
        )

        # 11 requests in <1 minute → exceeds the MINUTE window (10/min)
        for _ in range(10):
            await limiter.check(RateLimitPolicyName.CHAT, "user:internal:1")

        denied = await limiter.check(RateLimitPolicyName.CHAT, "user:internal:1")

        assert denied.allowed is False
        # The minute window is exceeded first → failing_rate = Rate(10, 60000)
        assert denied.limit == 10
        assert denied.retry_after_sec == 60

        await limiter.aclose()

    @pytest.mark.asyncio
    async def test_violation_fallback_on_missing_bucket(self):
        """If bucket is not in cache, _resolve_violation falls back to tightest_limit."""
        limiter = PyrateRateLimiter(
            fakeredis.aioredis.FakeRedis(decode_responses=True),
            _policies(),
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
        )

        # Exhaust LOGIN limit
        await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")
        await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")

        # Manually evict the bucket to trigger fallback
        limiter._factory.buckets.clear()

        denied = await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")

        assert denied.allowed is False
        assert denied.limit == 2  # fallback to tightest_limit
        assert denied.retry_after_sec == 60  # fallback to policy.retry_after_sec

        await limiter.aclose()


class TestPyrateRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_up_to_limit_then_denies(self):
        limiter = PyrateRateLimiter(
            fakeredis.aioredis.FakeRedis(decode_responses=True),
            _policies(),
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
        )
        principal = "ip:10.0.0.1"

        assert (await limiter.check(RateLimitPolicyName.LOGIN, principal)).allowed is True
        assert (await limiter.check(RateLimitPolicyName.LOGIN, principal)).allowed is True

        denied = await limiter.check(RateLimitPolicyName.LOGIN, principal)
        assert denied.allowed is False
        assert denied.retry_after_sec == 60
        assert denied.limit == 2
        assert denied.degraded is False

        await limiter.aclose()

    @pytest.mark.asyncio
    async def test_principals_are_isolated(self):
        limiter = PyrateRateLimiter(
            fakeredis.aioredis.FakeRedis(decode_responses=True),
            _policies(),
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
        )

        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed is True
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed is True
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed is False

        # A different principal has its own window with the search policy limit.
        other = await limiter.check(RateLimitPolicyName.SEARCH, "user:internal:42")
        assert other.allowed is True
        assert other.limit == 3

        await limiter.aclose()

    @pytest.mark.asyncio
    async def test_sets_key_ttl_for_widest_window(self):
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        limiter = PyrateRateLimiter(
            redis, _policies(), key_prefix=_KEY_PREFIX, max_buckets=10
        )

        await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")

        ttl = await redis.ttl(f"{_KEY_PREFIX}login:ip:1")
        assert 0 < ttl <= 120  # 2x the single 60s window

        await limiter.aclose()

    @pytest.mark.asyncio
    async def test_unknown_policy_raises(self):
        limiter = PyrateRateLimiter(
            fakeredis.aioredis.FakeRedis(decode_responses=True),
            _policies(),
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
        )

        with pytest.raises(ValueError, match="Unknown rate limit policy"):
            await limiter.check("does-not-exist", "ip:1")  # type: ignore[arg-type]

        await limiter.aclose()

    @pytest.mark.asyncio
    async def test_fail_open_on_backend_error(self):
        limiter = PyrateRateLimiter(
            _BrokenRedis(),
            _policies(),
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
            fail_open=True,
        )

        decision = await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")

        assert decision.allowed is True
        assert decision.degraded is True

        await limiter.aclose()

    @pytest.mark.asyncio
    async def test_fail_closed_when_configured(self):
        limiter = PyrateRateLimiter(
            _BrokenRedis(),
            _policies(),
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
            fail_open=False,
        )

        decision = await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")

        assert decision.allowed is False
        assert decision.degraded is True
        assert decision.retry_after_sec == 60

        await limiter.aclose()

    def test_empty_policies_rejected(self):
        with pytest.raises(ValueError, match="policies must not be empty"):
            PyrateRateLimiter(
                _BrokenRedis(), {}, key_prefix=_KEY_PREFIX, max_buckets=10
            )

    def test_invalid_max_buckets_rejected(self):
        with pytest.raises(ValueError, match="max_buckets"):
            PyrateRateLimiter(
                _BrokenRedis(), _policies(), key_prefix=_KEY_PREFIX, max_buckets=0
            )


class _ExpireFailRedis:
    """Redis that works for script_load/evalsha but fails on expire."""

    def __init__(self):
        self._inner = fakeredis.aioredis.FakeRedis(decode_responses=True)

    async def script_load(self, *a, **kw):
        return await self._inner.script_load(*a, **kw)

    async def evalsha(self, *a, **kw):
        return await self._inner.evalsha(*a, **kw)

    async def expire(self, *a, **kw):
        raise RedisError("expire failed")

    async def zadd(self, *a, **kw):
        return await self._inner.zadd(*a, **kw)

    async def zremrangebyscore(self, *a, **kw):
        return await self._inner.zremrangebyscore(*a, **kw)

    async def zcard(self, *a, **kw):
        return await self._inner.zcard(*a, **kw)

    async def zrange(self, *a, **kw):
        return await self._inner.zrange(*a, **kw)

    async def delete(self, *a, **kw):
        return await self._inner.delete(*a, **kw)

    async def keys(self, *a, **kw):
        return await self._inner.keys(*a, **kw)

    async def flushall(self, *a, **kw):
        return await self._inner.flushall(*a, **kw)


class TestRefreshTtlException:
    """Cover limiter.py:106-107 — _refresh_ttl exception handler."""

    @pytest.mark.asyncio
    async def test_refresh_ttl_failure_does_not_break_allowed_path(self):
        limiter = PyrateRateLimiter(
            _ExpireFailRedis(),
            _policies(),
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
        )

        # First request: bucket created, try_acquire succeeds, _refresh_ttl fails
        decision = await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")

        assert decision.allowed is True
        assert decision.degraded is False  # not degraded — TTL failure is best-effort

        await limiter.aclose()


class TestDoubleCheckedLock:
    """Cover bucket_factory.py:63-64 — second check inside _creation_lock."""

    @pytest.mark.asyncio
    async def test_cached_bucket_returned_inside_lock(self):
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        factory = RedisBucketFactory(
            redis, _policies(), key_prefix=_KEY_PREFIX, max_buckets=10
        )

        item = RateItem("login:ip:1", 1000, 1)

        # First call creates the bucket
        b1 = await factory.get(item)
        assert b1 is not None

        # Second call should hit the cached path inside the lock
        b2 = await factory.get(item)
        assert b2 is b1  # same object — cache hit

        assert len(factory.buckets) == 1


class TestLruEviction:
    """Cover bucket_factory.py:91-92 — _evict_overflow loop body."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_when_max_buckets_reached(self):
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        factory = RedisBucketFactory(
            redis, _policies(), key_prefix=_KEY_PREFIX, max_buckets=2
        )

        # Create 3 buckets → 3rd should evict the 1st
        for i in range(3):
            item = RateItem(f"login:ip:{i}", 1000 + i, 1)
            await factory.get(item)

        assert len(factory.buckets) == 2
        # Oldest (ip:0) should be evicted
        assert "login:ip:0" not in factory.buckets
        assert "login:ip:1" in factory.buckets
        assert "login:ip:2" in factory.buckets


class TestResolvePolicyError:
    """Cover bucket_factory.py:86-87 — _resolve_policy ValueError."""

    @pytest.mark.asyncio
    async def test_unknown_policy_in_item_name_raises(self):
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        factory = RedisBucketFactory(
            redis, _policies(), key_prefix=_KEY_PREFIX, max_buckets=10
        )

        item = RateItem("unknown_policy:ip:1", 1000, 1)

        with pytest.raises(ValueError, match="Unknown rate limit policy"):
            await factory.get(item)


class TestResolveViolationException:
    """Cover limiter.py:133-135 — _resolve_violation exception handler."""

    @pytest.mark.asyncio
    async def test_exception_in_bucket_lookup_falls_back(self):
        limiter = PyrateRateLimiter(
            fakeredis.aioredis.FakeRedis(decode_responses=True),
            _policies(),
            key_prefix=_KEY_PREFIX,
            max_buckets=10,
        )

        # Corrupt buckets to force an exception in _resolve_violation's try block
        class _BoomDict:
            def get(self, *a, **kw):
                raise RuntimeError("bucket lookup exploded")

        limiter._factory.buckets = _BoomDict()  # type: ignore[assignment]

        policy = _policies()[RateLimitPolicyName.LOGIN]
        fallback = await limiter._resolve_violation("login:ip:1", policy, RateLimitPolicyName.LOGIN)

        # Falls back to tightest_limit / policy.retry_after_sec
        assert fallback == (2, 60)

        await limiter.aclose()
