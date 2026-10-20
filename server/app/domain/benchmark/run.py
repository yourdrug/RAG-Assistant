"""BenchmarkRun domain entity — single benchmark run with config snapshot + metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from domain.benchmark.question import DEFAULT_DATASET


@dataclass
class BenchmarkRun:
    config_json: dict = field(default_factory=dict)
    summary_metrics: dict = field(default_factory=dict)
    duration_sec: float = 0.0
    llm_evaluated: bool = False
    dataset: str = DEFAULT_DATASET
    sweep_id: int | None = None
    per_question_results: list[dict] | None = None
    job_id: int | None = None
    legacy_hash: str | None = None
    artifacts: dict = field(default_factory=dict)
    filename: str | None = None
    id: int | None = None
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))
