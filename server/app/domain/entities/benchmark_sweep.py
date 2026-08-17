"""BenchmarkSweep domain entity — multi-config parameter search job."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from domain.value_objects.benchmark_strategy import BenchmarkStrategy
from domain.value_objects.sweep_status import BenchmarkSweepStatus


@dataclass
class BenchmarkSweep:
    strategy: str = BenchmarkStrategy.GRID.value
    search_space: dict = field(default_factory=dict)
    objective_weights: dict = field(default_factory=dict)
    dataset: str = "main"
    top_n_llm: int = 3
    status: str = BenchmarkSweepStatus.PENDING.value
    job_id: int | None = None
    total_configs: int = 0
    evaluated_configs: int = 0
    best_run_id: int | None = None
    id: int | None = None
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
