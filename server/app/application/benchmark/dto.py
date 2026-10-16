"""Application DTOs for Benchmark Lab — config apply and regression check results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from domain.value_objects.benchmark_dataset import BenchmarkDataset
from domain.value_objects.benchmark_scoring import DEFAULT_OBJECTIVE_WEIGHTS
from domain.value_objects.sweep_evaluation_mode import SweepEvaluationMode


if TYPE_CHECKING:
    from datetime import datetime

    from application.benchmark.ports import BenchmarkCheckpoints
    from application.chat import ChatRAGPort
    from domain.entities.benchmark_run import BenchmarkRun


@dataclass(frozen=True)
class BenchmarkQuestionCreateDTO:
    """DTO for creating a benchmark question — replaces presentation schema dependency."""

    question: str
    expected_answer: str | None = None
    source_hint: str | None = None
    annotations: dict[str, Any] | None = None
    tags: list[str] | None = None
    dataset: str = "main"
    notes: str | None = None
    is_active: bool = True


@dataclass(frozen=True)
class SweepCreateDTO:
    """DTO for creating a benchmark sweep — replaces presentation schema dependency."""

    strategy: str = "grid"
    search_space: dict[str, Any] = field(default_factory=dict)
    objective_weights: dict[str, Any] = field(default_factory=DEFAULT_OBJECTIVE_WEIGHTS.copy)
    dataset: str = "main"
    top_n_llm: int = 3
    judge_model: str | None = None
    evaluation_mode: str = SweepEvaluationMode.FAST.value


@dataclass(frozen=True)
class ApplyConfigResult:
    """Result of applying a benchmark run's config to the live system."""

    applied: int
    keys: list[str]
    failed: list[dict[str, Any]]


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
    run_id: int | None = None
    baseline_run_id: int | None = None
    results: list[RegressionCheckResult] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkResultSummary:
    id: int
    config_json: dict[str, Any] = field(default_factory=dict)
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0
    llm_evaluated: bool = False
    dataset: str = BenchmarkDataset.MAIN.value
    sweep_id: int | None = None
    creation_date: datetime | None = None


@dataclass(frozen=True)
class BenchmarkResultDetail:
    id: int
    summary: BenchmarkResultSummary
    per_question_results: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class BenchmarkResultsList:
    results: list[BenchmarkResultSummary] = field(default_factory=list)
    total: int = 0


@dataclass(frozen=True)
class BenchmarkImportResult:
    imported: int
    duplicates: int
    dry_run: bool


type RunCompareResult = tuple[list[BenchmarkRun], dict[str, list[dict[str, Any]]]]


class BenchmarkRunOptions(TypedDict, total=False):
    seed: int | None
    n_runs: int
    max_concurrent: int | None
    resume: bool
    checkpoints: BenchmarkCheckpoints | None
    checkpoint_prefix: str | None


class BenchmarkRunnerOptions(BenchmarkRunOptions, total=False):
    rag_service: ChatRAGPort | None
    export_files: bool
