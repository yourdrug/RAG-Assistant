"""Dynamic quotas retain shared Redis history across policy generations."""

import asyncio
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from redis.exceptions import RedisError

from config import Settings
from domain.value_objects.dynamic_config import DYNAMIC_FIELDS_MAP
from application.ports.rate_limit import RateLimitPolicyName
from infrastructure.rate_limit import limiter as module


@pytest.fixture
def configuration():
    return Settings(rate_limit_login_per_minute=3)


def make_limiter(redis, configuration):
    return module.DynamicRateLimiter(redis, configuration)


@pytest.mark.asyncio
async def test_lower_and_raise_quota_preserve_history(configuration):
    redis = fakeredis.aioredis.FakeRedis()
    limiter = make_limiter(redis, configuration)
    try:
        for _ in range(2):
            assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
        configuration.rate_limit_login_per_minute = 1
        assert not (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
        configuration.rate_limit_login_per_minute = 3
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
        assert not (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
        assert await redis.zcard(f"{configuration.rate_limit_redis_prefix}login:ip:1") == 3
    finally:
        await limiter.aclose()
        await redis.aclose()


@pytest.mark.asyncio
async def test_two_processes_share_history_with_concurrent_acquire(configuration):
    redis = fakeredis.aioredis.FakeRedis()
    limiters = [make_limiter(redis, configuration) for _ in range(2)]
    try:
        decisions = await asyncio.gather(
            *[limiters[i % 2].check(RateLimitPolicyName.LOGIN, "ip:1") for i in range(20)]
        )
        assert sum(d.allowed for d in decisions) == 3
        configuration.rate_limit_login_per_minute = 4
        decisions = await asyncio.gather(
            *[limiters[i % 2].check(RateLimitPolicyName.LOGIN, "ip:1") for i in range(20)]
        )
        assert sum(d.allowed for d in decisions) == 1
    finally:
        for limiter in limiters:
            await limiter.aclose()
        await redis.aclose()


@pytest.mark.asyncio
async def test_disabled_and_fail_open_are_live(configuration):
    redis = AsyncMock()
    redis.script_load.side_effect = RedisError("offline")
    limiter = make_limiter(redis, configuration)
    try:
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
        configuration.rate_limit_fail_open = False
        assert not (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
        configuration.rate_limit_enabled = False
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
        configuration.rate_limit_enabled = True
        assert not (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
    finally:
        await limiter.aclose()


def test_quotas_are_dynamic_and_positive():
    fields = [key for key in Settings.model_fields if key.startswith("rate_limit_") and "_per_" in key]
    for key in fields:
        assert key in DYNAMIC_FIELDS_MAP
        with pytest.raises(ValueError):
            Settings(**{key: 0})
    assert {"rate_limit_enabled", "rate_limit_fail_open"} <= DYNAMIC_FIELDS_MAP.keys()


@pytest.mark.asyncio
async def test_rotation_during_acquire_keeps_old_generation_alive(configuration):
    entered, resume = asyncio.Event(), asyncio.Event()

    class PausedRedis(fakeredis.aioredis.FakeRedis):
        async def evalsha(self, *args, **kwargs):
            if not entered.is_set():
                entered.set()
                await resume.wait()
            return await super().evalsha(*args, **kwargs)

    redis = PausedRedis()
    limiter = make_limiter(redis, configuration)
    old_request = asyncio.create_task(limiter.check(RateLimitPolicyName.LOGIN, "ip:1"))
    try:
        await entered.wait()
        configuration.rate_limit_login_per_minute = 1
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
        resume.set()
        assert (await old_request).allowed  # admitted using its original quota snapshot
        assert not (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed
        assert len(limiter._generations) == 1
    finally:
        resume.set()
        await old_request
        await limiter.aclose()
        await redis.aclose()
