"""Benchmark schemas (runner + lab: questions, sweeps, runs, history)."""

from __future__ import annotations

from domain.value_objects.benchmark_scoring import DEFAULT_OBJECTIVE_WEIGHTS, validate_objective_weights

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from domain.value_objects.benchmark_dataset import BenchmarkDataset
from domain.value_objects.benchmark_annotations import validate_annotations
from domain.value_objects.benchmark_strategy import BenchmarkStrategy
from domain.value_objects.sweep_evaluation_mode import SweepEvaluationMode, validate_sweep_mode


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(default=BenchmarkDataset.MAIN.value, min_length=1, max_length=100)
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
    per_question_results: list[dict] | None = None


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
    annotations: dict | None = None

    _validate_annotations = field_validator("annotations")(validate_annotations)
    tags: list[str] | None = None
    dataset: str = BenchmarkDataset.MAIN.value
    is_active: bool = True
    notes: str | None = None


class BenchmarkQuestionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str | None = None
    expected_answer: str | None = None
    source_hint: str | None = None
    annotations: dict | None = None

    _validate_annotations = field_validator("annotations")(validate_annotations)
    tags: list[str] | None = None
    dataset: str | None = None
    is_active: bool | None = None
    notes: str | None = None


class BenchmarkQuestionResponse(BaseModel):
    id: int
    question: str
    expected_answer: str | None = None
    source_hint: str | None = None
    annotations: dict | None = None

    _validate_annotations = field_validator("annotations")(validate_annotations)
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

    strategy: BenchmarkStrategy = Field(
        BenchmarkStrategy.GRID,
        description="Search strategy. successive_halving is a deprecated alias for optuna_tpe.",
    )
    search_space: dict
    objective_weights: dict[str, float] = Field(default_factory=lambda: DEFAULT_OBJECTIVE_WEIGHTS.copy())
    dataset: str = BenchmarkDataset.MAIN.value
    top_n_llm: int = Field(3, ge=0, le=20)
    evaluation_mode: SweepEvaluationMode = SweepEvaluationMode.FAST
    judge_model: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("strategy")
    @classmethod
    def normalize_strategy(cls, value: BenchmarkStrategy) -> BenchmarkStrategy:
        if value == BenchmarkStrategy.SUCCESSIVE_HALVING:
            return BenchmarkStrategy.OPTUNA_TPE
        return value

    @field_validator("judge_model")
    @classmethod
    def validate_judge_model(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("judge_model must not be blank")
        return value

    @field_validator("objective_weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        return validate_objective_weights(value, require_retrieval=False)

    @model_validator(mode="after")
    def validate_evaluation_mode(self) -> SweepCreateRequest:
        validate_sweep_mode(self.evaluation_mode, self.strategy)
        self.objective_weights = validate_objective_weights(
            self.objective_weights, require_retrieval=self.evaluation_mode == SweepEvaluationMode.FAST
        )
        return self


class SweepResponse(BaseModel):
    id: int
    status: str
    strategy: str
    search_space: dict
    objective_weights: dict
    dataset: str
    top_n_llm: int
    evaluation_mode: SweepEvaluationMode = SweepEvaluationMode.FAST
    judge_model: str | None = None
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
