"""Tests for the rate limit policy catalog built from settings."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from application.ports.rate_limit import RateLimitPolicyName, RateSpec  # noqa: E402
from config import settings  # noqa: E402
from infrastructure.rate_limit.policies import build_policies  # noqa: E402
from pyrate_limiter import InMemoryBucket, Rate  # noqa: E402


def _stub_settings() -> SimpleNamespace:
    return SimpleNamespace(
        rate_limit_login_per_minute=5,
        rate_limit_chat_per_minute=10,
        rate_limit_chat_per_day=100,
        rate_limit_upload_per_hour=20,
        rate_limit_search_per_minute=60,
        rate_limit_write_per_minute=60,
        rate_limit_benchmark_per_hour=3,
    )


class TestBuildPolicies:
    def test_all_policy_names_covered(self):
        policies = build_policies(_stub_settings())
        assert set(policies) == set(RateLimitPolicyName)

    def test_exact_default_windows(self):
        policies = build_policies(_stub_settings())
        assert policies[RateLimitPolicyName.LOGIN].rates == (RateSpec(5, 60_000),)
        assert policies[RateLimitPolicyName.CHAT].rates == (
            RateSpec(10, 60_000),
            RateSpec(100, 86_400_000),
        )
        assert policies[RateLimitPolicyName.UPLOAD].rates == (RateSpec(20, 3_600_000),)
        assert policies[RateLimitPolicyName.SEARCH].rates == (RateSpec(60, 60_000),)
        assert policies[RateLimitPolicyName.WRITE].rates == (RateSpec(60, 60_000),)
        assert policies[RateLimitPolicyName.BENCHMARK].rates == (RateSpec(3, 3_600_000),)

    def test_policy_name_matches_key(self):
        for name, policy in build_policies(_stub_settings()).items():
            assert policy.name is name

    def test_retry_after_is_positive(self):
        for policy in build_policies(_stub_settings()).values():
            assert policy.retry_after_sec > 0

    @pytest.mark.parametrize("policy_name", list(RateLimitPolicyName))
    def test_rate_lists_are_valid_for_pyrate_limiter(self, policy_name):
        """pyrate-limiter rejects ill-ordered rates at bucket construction."""
        policy = build_policies(_stub_settings())[policy_name]
        rates = [Rate(spec.limit, spec.interval_ms) for spec in policy.rates]
        InMemoryBucket(rates)  # raises ValueError on ill-formed rate list

    def test_builds_from_real_settings(self):
        policies = build_policies(settings)
        assert policies[RateLimitPolicyName.CHAT].rates[0].limit == settings.rate_limit_chat_per_minute
