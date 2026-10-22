"""Per-principal Redis bucket factory for pyrate-limiter.

Each item name (``{policy}:{principal}``) maps to its own ``RedisBucket`` backed
by a dedicated Redis sorted set, so principals never share a window. Buckets are
cached in a bounded LRU; eviction drops only the process-local bucket.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from time import time_ns
from typing import TYPE_CHECKING

from pyrate_limiter import BucketFactory, Rate, RateItem, RedisBucket

from application.ports.rate_limit import RateLimitPolicy, RateLimitPolicyName

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger("default")

_MS_PER_SECOND = 1000
_TTL_WINDOW_MULTIPLIER = 2  # key TTL = 2x the widest policy window

# pyrate-limiter 4.5.0's RedisBucket reply/argument ABI, with admission time
# captured atomically in Redis. Client timestamps can precede concurrent writes.
SERVER_TIME_ADMISSION = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
local shift = now - tonumber(ARGV[1])
local weight = tonumber(ARGV[2])
for i = 1, tonumber(ARGV[4]) do
    local offset = (i - 1) * 3
    local window_start = tonumber(ARGV[5 + offset]) + shift
    local limit = tonumber(ARGV[6 + offset])
    local rank = tonumber(ARGV[7 + offset])
    local count = redis.call('ZCOUNT', KEYS[1], window_start, now)
    if count + weight > limit then
        local blocking_timestamp = -1
        if rank >= 0 then
            local blocking = redis.call('ZRANGE', KEYS[1], -1 - rank, -1 - rank, 'WITHSCORES')
            if blocking[2] then blocking_timestamp = tonumber(blocking[2]) end
        end
        return {i - 1, blocking_timestamp}
    end
end
for i = 1, weight do
    redis.call('ZADD', KEYS[1], now, ARGV[3] .. i)
end
return {-1, -1}
"""


class ServerTimedRedisBucket(RedisBucket):
    async def now(self):
        # Background expiry must use the same clock as admission across hosts.
        seconds, microseconds = await self.redis.time()
        return seconds * 1000 + microseconds // 1000


class RedisBucketFactory(BucketFactory):
    """Creates and caches one RedisBucket per ``{policy}:{principal}`` item name."""

    def __init__(
        self,
        redis: Redis,
        policies: dict[RateLimitPolicyName, RateLimitPolicy],
        *,
        key_prefix: str,
        max_buckets: int,
    ) -> None:
        if max_buckets < 1:
            raise ValueError(f"max_buckets must be >= 1, got {max_buckets}")

        self.buckets: OrderedDict[str, RedisBucket] = OrderedDict()

        self._redis = redis
        self._policies = policies
        self._key_prefix = key_prefix
        self._max_buckets = max_buckets
        self._creation_lock = asyncio.Lock()

    def wrap_item(self, name: str, weight: int = 1) -> RateItem:
        return RateItem(name, time_ns() // 1_000_000, weight)

    async def get(self, item: RateItem) -> RedisBucket:
        cached = self.buckets.get(item.name)

        if cached is not None:
            self.buckets.move_to_end(item.name)
            return cached

        async with self._creation_lock:
            cached = self.buckets.get(item.name)

            if cached is not None:
                self.buckets.move_to_end(item.name)
                return cached

            policy = self._resolve_policy(item.name)
            key = f"{self._key_prefix}{item.name}"
            rates = [Rate(spec.limit, spec.interval_ms) for spec in policy.rates]
            script_hash = await self._redis.script_load(SERVER_TIME_ADMISSION)
            bucket = ServerTimedRedisBucket(rates, self._redis, key, script_hash)
            self.schedule_leak(bucket)
            self.buckets[item.name] = bucket
            await self._evict_overflow()
            return bucket

    async def aclose(self) -> None:
        task = self._leaker.aio_leak_task if self._leaker is not None else None
        self.close()

        if task is not None:
            task.cancel()
            result = (await asyncio.gather(task, return_exceptions=True))[0]

            if isinstance(result, Exception):
                logger.error(
                    "Rate limit background expiry failed before shutdown",
                    exc_info=(type(result), result, result.__traceback__),
                )

    async def refresh_key_ttl(self, item_name: str) -> None:
        """Keep an active principal's key alive for the widest policy window."""
        policy = self._resolve_policy(item_name)
        widest_ms = max(spec.interval_ms for spec in policy.rates)
        ttl_sec = max(1, (widest_ms * _TTL_WINDOW_MULTIPLIER) // _MS_PER_SECOND)
        await self._redis.expire(f"{self._key_prefix}{item_name}", ttl_sec)

    def _resolve_policy(self, item_name: str) -> RateLimitPolicy:
        policy_name = item_name.partition(":")[0]
        try:
            return self._policies[RateLimitPolicyName(policy_name)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unknown rate limit policy in item name: {item_name!r}") from exc

    async def _evict_overflow(self) -> None:
        while len(self.buckets) > self._max_buckets:
            item_name, bucket = self.buckets.popitem(last=False)
            self.dispose(bucket)
