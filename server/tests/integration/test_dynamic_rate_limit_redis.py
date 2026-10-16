"""Two OS processes retain quota history across dynamic policy changes."""

import asyncio
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from uuid import uuid4

import pytest
from redis.asyncio import Redis

from application.ports.rate_limit import RateLimitPolicyName
from config import Settings
from infrastructure.rate_limit.limiter import DynamicRateLimiter


def acquire_in_process(url, prefix, quota, enabled=True):
    async def acquire():
        redis = Redis.from_url(url)
        limiter = DynamicRateLimiter(
            redis,
            Settings(
                rate_limit_enabled=enabled,
                rate_limit_fail_open=False,
                rate_limit_login_per_minute=quota,
                rate_limit_redis_prefix=prefix,
            ),
        )
        try:
            return sum([(await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed for _ in range(2)])
        finally:
            await limiter.aclose()
            await redis.aclose()

    return asyncio.run(acquire())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_processes_preserve_history_when_quota_changes():
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.skip("Set TEST_REDIS_URL to an isolated test Redis")
    prefix = f"test:dynamic-rate:{uuid4().hex}:"
    redis = Redis.from_url(url)
    key = f"{prefix}login:ip:1"
    loop = asyncio.get_running_loop()
    try:
        context = multiprocessing.get_context("spawn")
        with (
            ProcessPoolExecutor(max_workers=1, mp_context=context) as first,
            ProcessPoolExecutor(max_workers=1, mp_context=context) as second,
        ):

            async def batch(quota, enabled=True):
                return sum(
                    await asyncio.gather(
                        *(
                            loop.run_in_executor(pool, acquire_in_process, url, prefix, quota, enabled)
                            for pool in (first, second)
                        )
                    )
                )

            assert await batch(3) == 3
            assert await batch(2) == 0
            assert await batch(5) == 2
            assert await batch(5, enabled=False) == 4
            assert await batch(5) == 0
        assert await redis.zcard(key) == 5
    finally:
        await redis.delete(key)
        await redis.aclose()
