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
    return {
        RateLimitPolicyName.LOGIN: RateLimitPolicy(
            name=RateLimitPolicyName.LOGIN,
            rates=(RateSpec(settings.rate_limit_login_per_minute, _MINUTE_MS),),
            retry_after_sec=60,
        ),
        RateLimitPolicyName.CHAT: RateLimitPolicy(
            name=RateLimitPolicyName.CHAT,
            rates=(
                RateSpec(settings.rate_limit_chat_per_minute, _MINUTE_MS),
                RateSpec(settings.rate_limit_chat_per_day, _DAY_MS),
            ),
            retry_after_sec=60,
        ),
        RateLimitPolicyName.UPLOAD: RateLimitPolicy(
            name=RateLimitPolicyName.UPLOAD,
            rates=(RateSpec(settings.rate_limit_upload_per_hour, _HOUR_MS),),
            retry_after_sec=3600,
        ),
        RateLimitPolicyName.SEARCH: RateLimitPolicy(
            name=RateLimitPolicyName.SEARCH,
            rates=(RateSpec(settings.rate_limit_search_per_minute, _MINUTE_MS),),
            retry_after_sec=60,
        ),
        RateLimitPolicyName.WRITE: RateLimitPolicy(
            name=RateLimitPolicyName.WRITE,
            rates=(RateSpec(settings.rate_limit_write_per_minute, _MINUTE_MS),),
            retry_after_sec=60,
        ),
        RateLimitPolicyName.BENCHMARK: RateLimitPolicy(
            name=RateLimitPolicyName.BENCHMARK,
            rates=(RateSpec(settings.rate_limit_benchmark_per_hour, _HOUR_MS),),
            retry_after_sec=3600,
        ),
    }
