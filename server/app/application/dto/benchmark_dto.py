"""Application DTOs for Benchmark Lab — config apply and regression check results."""

from __future__ import annotations

from domain.value_objects.benchmark_scoring import DEFAULT_OBJECTIVE_WEIGHTS

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkQuestionCreateDTO:
    """DTO for creating a benchmark question — replaces presentation schema dependency."""

    question: str
    expected_answer: str | None = None
    source_hint: str | None = None
    annotations: dict | None = None
    tags: list[str] | None = None
    dataset: str = "main"
    notes: str | None = None
    is_active: bool = True


@dataclass(frozen=True)
class SweepCreateDTO:
    """DTO for creating a benchmark sweep — replaces presentation schema dependency."""

    strategy: str = "grid"
    search_space: dict = field(default_factory=dict)
    objective_weights: dict = field(default_factory=lambda: DEFAULT_OBJECTIVE_WEIGHTS.copy())
    dataset: str = "main"
    top_n_llm: int = 3


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
