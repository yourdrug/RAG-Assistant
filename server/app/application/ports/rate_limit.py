"""Rate limiting ports and policy vocabulary (framework-agnostic).

Presentation knows only policy names; infrastructure owns concrete rate
values, the distributed backend, and the enforcement mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class RateLimitPolicyName(StrEnum):
    """Named quota policies referenced by routes."""

    LOGIN = "login"
    CHAT = "chat"
    UPLOAD = "upload"
    SEARCH = "search"
    WRITE = "write"
    BENCHMARK = "benchmark"


@dataclass(frozen=True)
class RateSpec:
    """One quota window, backend-agnostic (interval in milliseconds)."""

    limit: int
    interval_ms: int


@dataclass(frozen=True)
class RateLimitPolicy:
    """A named set of windows enforced for a single principal."""

    name: RateLimitPolicyName
    rates: tuple[RateSpec, ...]
    retry_after_sec: int


@dataclass(frozen=True)
class RateLimitDecision:
    """Verdict of a single limiter check.

    ``limit`` is the capacity of the tightest window (used for the
    ``X-RateLimit-Limit`` header); ``retry_after_sec`` is a policy-level hint.
    ``degraded`` means the backend was unavailable and fail-open applied.
    """

    allowed: bool
    retry_after_sec: int = 0
    limit: int = 0
    degraded: bool = False


class RateLimiterPort(Protocol):
    """Application-facing port for rate limiting (implemented in infrastructure)."""

    async def check(self, policy: RateLimitPolicyName, principal: str) -> RateLimitDecision: ...

    async def aclose(self) -> None: ...
