"""Build rate-limit policies from application settings.

Single source of truth: routes reference policy names, all numbers live here
(and in ``Settings``).
"""

from __future__ import annotations

from application.ports.rate_limit import RateLimitPolicy, RateLimitPolicyName, RateSpec
from config import Settings

_MINUTE_MS = 60_000
_HOUR_MS = 60 * _MINUTE_MS
_DAY_MS = 24 * _HOUR_MS


def build_policies(settings: Settings) -> dict[RateLimitPolicyName, RateLimitPolicy]:
    """Map settings to the full policy catalog."""
    catalog = (
        (RateLimitPolicyName.LOGIN, ((settings.rate_limit_login_per_minute, _MINUTE_MS),), 60),
        (
            RateLimitPolicyName.CHAT,
            ((settings.rate_limit_chat_per_minute, _MINUTE_MS), (settings.rate_limit_chat_per_day, _DAY_MS)),
            60,
        ),
        (RateLimitPolicyName.UPLOAD, ((settings.rate_limit_upload_per_hour, _HOUR_MS),), 3600),
        (RateLimitPolicyName.SEARCH, ((settings.rate_limit_search_per_minute, _MINUTE_MS),), 60),
        (RateLimitPolicyName.WRITE, ((settings.rate_limit_write_per_minute, _MINUTE_MS),), 60),
        (RateLimitPolicyName.BENCHMARK, ((settings.rate_limit_benchmark_per_hour, _HOUR_MS),), 3600),
    )
    return {
        name: RateLimitPolicy(
            name=name,
            rates=tuple(RateSpec(limit, interval) for limit, interval in rates),
            retry_after_sec=retry_after,
        )
        for name, rates, retry_after in catalog
    }
