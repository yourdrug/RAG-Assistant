"""Benchmark schemas (runner + lab: questions, sweeps, runs, history)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from domain.value_objects.benchmark_dataset import BenchmarkDataset
from domain.value_objects.benchmark_strategy import BenchmarkStrategy


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions_path: str | None = None
    out_dir: str | None = None
    top_k: int | None = None
    judge_model: str | None = None


class BenchmarkResponse(BaseModel):
    status: str


class BenchmarkResultSummary(BaseModel):
    id: int
    config_json: dict
    summary_metrics: dict
    duration_sec: float
    llm_evaluated: bool
    dataset: str
    sweep_id: int | None = None
    creation_date: datetime | None = None


class BenchmarkResultDetail(BaseModel):
    id: int
    summary: BenchmarkResultSummary
    per_question_results: dict | None = None


class BenchmarkResultsListResponse(BaseModel):
    results: list[BenchmarkResultSummary]
    total: int


# ---------------------------------------------------------------------------
# Benchmark Lab — Questions
# ---------------------------------------------------------------------------


class BenchmarkQuestionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=5000)
    expected_answer: str | None = None
    source_hint: str | None = None
    tags: list[str] | None = None
    dataset: str = BenchmarkDataset.MAIN.value
    notes: str | None = None


class BenchmarkQuestionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = None
    expected_answer: str | None = None
    source_hint: str | None = None
    tags: list[str] | None = None
    dataset: str | None = None
    is_active: bool | None = None
    notes: str | None = None


class BenchmarkQuestionResponse(BaseModel):
    id: int
    question: str
    expected_answer: str | None = None
    source_hint: str | None = None
    tags: list[str] | None = None
    dataset: str
    is_active: bool
    created_by: int | None = None
    notes: str | None = None
    creation_date: datetime | None = None


class BenchmarkQuestionsListResponse(BaseModel):
    questions: list[BenchmarkQuestionResponse]
    total: int


class BenchmarkQuestionsImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[BenchmarkQuestionCreate] = Field(..., max_length=500)


class BenchmarkQuestionsImportResponse(BaseModel):
    imported: int


# ---------------------------------------------------------------------------
# Benchmark Lab — Sweeps
# ---------------------------------------------------------------------------


class SweepCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(BenchmarkStrategy.GRID.value, pattern="^(grid|random|successive_halving)$")
    search_space: dict
    objective_weights: dict = Field(
        default_factory=lambda: {"hit_rate": 0.4, "faithfulness": 0.3, "relevancy": 0.3}
    )
    dataset: str = BenchmarkDataset.MAIN.value
    top_n_llm: int = Field(3, ge=0, le=20)


class SweepResponse(BaseModel):
    id: int
    status: str
    strategy: str
    search_space: dict
    objective_weights: dict
    dataset: str
    top_n_llm: int
    total_configs: int
    evaluated_configs: int
    best_run_id: int | None = None
    job_id: int | None = None
    creation_date: datetime | None = None


class SweepsListResponse(BaseModel):
    sweeps: list[SweepResponse]
    total: int


# ---------------------------------------------------------------------------
# Benchmark Lab — Runs
# ---------------------------------------------------------------------------


class BenchmarkRunResponse(BaseModel):
    id: int
    sweep_id: int | None = None
    config_json: dict
    summary_metrics: dict
    duration_sec: float
    llm_evaluated: bool
    dataset: str
    filename: str | None = None
    creation_date: datetime | None = None


class BenchmarkRunsListResponse(BaseModel):
    runs: list[BenchmarkRunResponse]
    total: int


class RunApplyFailed(BaseModel):
    key: str
    error: str


class RunApplyResponse(BaseModel):
    applied: int
    keys: list[str]
    failed: list[RunApplyFailed]


class RunCompareResponse(BaseModel):
    runs: list[BenchmarkRunResponse]
    diff: dict


class RegressionCheckResult(BaseModel):
    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    threshold: float
    failed: bool
    note: str | None = None


class RegressionCheckResponse(BaseModel):
    passed: bool
    results: list[RegressionCheckResult]


# ---------------------------------------------------------------------------
# Benchmark Lab — History / Trends
# ---------------------------------------------------------------------------


class BenchmarkHistoryPoint(BaseModel):
    run_id: int
    creation_date: datetime | None = None
    metrics: dict
    config_summary: dict
    dataset: str
    llm_evaluated: bool


class BenchmarkHistoryResponse(BaseModel):
    points: list[BenchmarkHistoryPoint]
    total: int


class BenchmarkDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deleted: bool


class BenchmarkCancelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cancelled: bool


class SourceFilesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[str]
