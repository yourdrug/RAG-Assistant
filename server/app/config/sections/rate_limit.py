"""Request and cost rate limiting."""

from __future__ import annotations

from pydantic import BaseModel, Field

from config.ownership import Static


class RateLimitSection(BaseModel):
    rate_limit_enabled: bool = True
    rate_limit_fail_open: bool = True  # Redis unavailable → allow traffic (with metric)
    rate_limit_redis_prefix: Static[str] = "rag:ratelimit:v1:"
    rate_limit_login_per_minute: int = Field(default=5, ge=1)
    rate_limit_chat_per_minute: int = Field(default=10, ge=1)
    rate_limit_chat_per_day: int = Field(default=100, ge=1)
    rate_limit_upload_per_hour: int = Field(default=20, ge=1)
    rate_limit_search_per_minute: int = Field(default=60, ge=1)
    rate_limit_write_per_minute: int = Field(default=60, ge=1)
    rate_limit_benchmark_per_hour: int = Field(default=60, ge=1)
    rate_limit_max_buckets: Static[int] = 10000  # LRU cap for per-principal buckets


class CostLimitSection(BaseModel):
    cost_rate_limit_enabled: bool = False
    cost_hourly_limit: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    cost_daily_limit: float = Field(default=5.0, gt=0, allow_inf_nan=False)
