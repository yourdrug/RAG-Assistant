"""Application DTOs for Benchmark Lab — config apply and regression check results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ApplyConfigResult:
    """Result of applying a benchmark run's config to the live system."""

    applied: int
    keys: list[str]
    failed: list[dict]


@dataclass(frozen=True)
class RegressionCheckResult:
    """Single metric regression check result."""

    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    threshold: float
    failed: bool
    note: str | None = None


@dataclass(frozen=True)
class RegressionCheckOutput:
    """Full regression check output."""

    passed: bool
    results: list[RegressionCheckResult] = field(default_factory=list)
