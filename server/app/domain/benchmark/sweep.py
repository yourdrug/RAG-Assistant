"""Sweep configuration, evaluation modes and execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from domain.benchmark.question import DEFAULT_DATASET
from domain.benchmark.scoring import DEFAULT_OBJECTIVE_WEIGHTS
from domain.shared.enums import LifecycleStatus, parse_enum
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError


class BenchmarkStrategy(StrEnum):
    GRID = "grid"
    RANDOM = "random"
    OPTUNA_TPE = "optuna_tpe"


class BenchmarkSweepStatus(
    LifecycleStatus,
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
    resolved = parse_enum(SweepEvaluationMode, mode)
    strategy = parse_enum(BenchmarkStrategy, strategy)

    if resolved == SweepEvaluationMode.FULL and strategy != BenchmarkStrategy.GRID:
        msg = "Full evaluation requires grid search to evaluate every combination"
        raise ValidationError(msg)

    return resolved


@dataclass
class BenchmarkSweep:
    strategy: BenchmarkStrategy = BenchmarkStrategy.GRID
    search_space: dict[str, Any] = field(default_factory=dict)
    objective_weights: dict[str, Any] = field(default_factory=DEFAULT_OBJECTIVE_WEIGHTS.copy)
    dataset: str = DEFAULT_DATASET
    top_n_llm: int = 3
    judge_model: str | None = None
    evaluation_mode: SweepEvaluationMode = SweepEvaluationMode.FAST
    status: BenchmarkSweepStatus = BenchmarkSweepStatus.PENDING
    job_id: int | None = None
    total_configs: int = 0
    evaluated_configs: int = 0
    best_run_id: int | None = None
    id: int | None = None
    version: int = 0
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.strategy = parse_enum(BenchmarkStrategy, self.strategy)
        self.evaluation_mode = parse_enum(SweepEvaluationMode, self.evaluation_mode)
        self.status = parse_enum(BenchmarkSweepStatus, self.status)

    def transition_to(self, status: BenchmarkSweepStatus) -> None:
        allowed = {
            BenchmarkSweepStatus.PENDING: {
                BenchmarkSweepStatus.RUNNING,
                BenchmarkSweepStatus.FAILED,
                BenchmarkSweepStatus.CANCELLED,
            },
            BenchmarkSweepStatus.RUNNING: {
                BenchmarkSweepStatus.DONE,
                BenchmarkSweepStatus.FAILED,
                BenchmarkSweepStatus.CANCELLED,
            },
            BenchmarkSweepStatus.DONE: {BenchmarkSweepStatus.PENDING},
            BenchmarkSweepStatus.FAILED: {BenchmarkSweepStatus.PENDING},
            BenchmarkSweepStatus.CANCELLED: {BenchmarkSweepStatus.PENDING},
        }

        if status != self.status and status not in allowed[self.status]:
            message = f"Cannot transition sweep from {self.status} to {status}"
            raise BusinessRuleViolationError(message)

        self.status = status
