"""Integration test: rate limiter against a real Redis (Lua/EVALSHA path).

Requires a reachable Redis; skipped otherwise. Point the test at any instance::

    RATE_LIMIT_TEST_REDIS_URL=redis://localhost:6379/15 \
        uv run pytest -m integration tests/test_rate_limit_redis_integration.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import redis.asyncio as aioredis
from redis.exceptions import RedisError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from application.ports.rate_limit import (  # noqa: E402
    RateLimitPolicy,
    RateLimitPolicyName,
    RateSpec,
)
from infrastructure.rate_limit.limiter import PyrateRateLimiter  # noqa: E402

_DEFAULT_URL = "redis://localhost:6379/15"


def _policies() -> dict[RateLimitPolicyName, RateLimitPolicy]:
    return {
        RateLimitPolicyName.LOGIN: RateLimitPolicy(
            name=RateLimitPolicyName.LOGIN,
            rates=(RateSpec(2, 60_000),),
            retry_after_sec=60,
        )
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rate_limit_round_trip_on_real_redis():
    url = os.environ.get("RATE_LIMIT_TEST_REDIS_URL", _DEFAULT_URL)
    client = aioredis.from_url(url, decode_responses=True)
    try:
        await client.ping()
    except (RedisError, OSError) as exc:
        await client.aclose()
        pytest.skip(f"real Redis unavailable at {url}: {exc}")

    limiter = PyrateRateLimiter(client, _policies(), key_prefix="test:integration:rl:", max_buckets=10)
    try:
        await client.flushdb()
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed is True
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed is True
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:1")).allowed is False
        assert (await limiter.check(RateLimitPolicyName.LOGIN, "ip:2")).allowed is True
    finally:
        await limiter.aclose()
        await client.flushdb()
        await client.aclose()
