"""Request and cost rate limiting."""

from __future__ import annotations

from pydantic import BaseModel, Field, PositiveInt

from config.ownership import Static


class RateLimitSection(BaseModel):
    rate_limit_enabled: bool = True
    rate_limit_fail_open: bool = True  # Redis unavailable → allow traffic (with metric)
    rate_limit_redis_prefix: Static[str] = "rag:ratelimit:v1:"
    rate_limit_login_per_minute: PositiveInt = 5
    rate_limit_chat_per_minute: PositiveInt = 10
    rate_limit_chat_per_day: PositiveInt = 100
    rate_limit_upload_per_hour: PositiveInt = 20
    rate_limit_search_per_minute: PositiveInt = 60
    rate_limit_write_per_minute: PositiveInt = 60
    rate_limit_benchmark_per_hour: PositiveInt = 60
    rate_limit_max_buckets: Static[PositiveInt] = 10000  # LRU cap for per-principal buckets


class CostLimitSection(BaseModel):
    cost_rate_limit_enabled: bool = False
    cost_hourly_limit: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    cost_daily_limit: float = Field(default=5.0, gt=0, allow_inf_nan=False)
