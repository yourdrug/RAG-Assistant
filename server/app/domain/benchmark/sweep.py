"""Benchmark sweep: domain models and rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from domain.benchmark.question import DEFAULT_DATASET
from domain.benchmark.scoring import DEFAULT_OBJECTIVE_WEIGHTS
from domain.shared.enums import LifecycleStatusMixin


class BenchmarkStrategy(StrEnum):
    GRID = "grid"
    RANDOM = "random"
    OPTUNA_TPE = "optuna_tpe"
    # Deprecated input/persisted value; executes the same TPE search.
    SUCCESSIVE_HALVING = "successive_halving"


class BenchmarkSweepStatus(
    LifecycleStatusMixin,
    StrEnum,
    terminal=frozenset({"done", "failed", "cancelled"}),
    active=frozenset({"pending", "running"}),
):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SweepEvaluationMode(StrEnum):
    FAST = "fast"
    FULL = "full"


def validate_sweep_mode(mode: str, strategy: str) -> SweepEvaluationMode:
    resolved = SweepEvaluationMode(mode)
    if resolved == SweepEvaluationMode.FULL and strategy != BenchmarkStrategy.GRID:
        raise ValueError("Full evaluation requires grid search to evaluate every combination")
    return resolved


@dataclass
class BenchmarkSweep:
    strategy: str = BenchmarkStrategy.GRID.value
    search_space: dict = field(default_factory=dict)
    objective_weights: dict = field(default_factory=lambda: DEFAULT_OBJECTIVE_WEIGHTS.copy())
    dataset: str = DEFAULT_DATASET
    top_n_llm: int = 3
    judge_model: str | None = None
    evaluation_mode: str = SweepEvaluationMode.FAST.value
    status: str = BenchmarkSweepStatus.PENDING.value
    job_id: int | None = None
    total_configs: int = 0
    evaluated_configs: int = 0
    best_run_id: int | None = None
    id: int | None = None
    version: int = 0
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
