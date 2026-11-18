"""Benchmark acceptance policy, composition, noise, human labels and judge agreement."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AcceptanceThresholdsBody(BaseModel):
    min_fragment_recall_at_k: float | None = Field(default=None, ge=0, le=1)
    min_context_fragment_recall: float | None = Field(default=None, ge=0, le=1)
    min_correctness: float | None = Field(default=None, ge=0, le=10)
    max_false_answer_rate: float | None = Field(default=None, ge=0, le=1)


class DatasetPolicyResponse(BaseModel):
    dataset: str
    thresholds: AcceptanceThresholdsBody
    noise: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    updated_by: int | None = None
    update_date: datetime | None = None


class JudgeCalibrationResponse(BaseModel):
    id: int | None
    run_id: int | None
    judge_model: str | None = None
    judge_rubric_version: str | None = None
    # Per metric: n, mae, spearman, kappa and min_kappa.
    metrics: dict[str, dict[str, Any]]
    measured_at: datetime
    created_by: int | None = None


class JudgeCalibrationsResponse(BaseModel):
    dataset: str
    calibrations: list[JudgeCalibrationResponse]


class DatasetPoliciesResponse(BaseModel):
    policies: list[DatasetPolicyResponse]


class CompositionResponse(BaseModel):
    dataset: str
    total: int
    counts: dict[str, int]
    shares: dict[str, float]
    failures: list[str]
    passed: bool


class ReachabilityResponse(BaseModel):
    dataset: str
    space_id: int
    pack_key: str
    questions: int
    fragments: int
    reachable: int
    share: float | None
    # question_id, question, source, text (shortened) and reason: no_document | not_in_chunk.
    unreachable: list[dict[str, Any]]


class NoiseRequest(BaseModel):
    run_ids: list[int] = Field(min_length=3, max_length=10)


class HumanLabelBody(BaseModel):
    question_key: str = Field(min_length=1, max_length=200)
    metric: Literal["correctness", "faithfulness"]
    score: float = Field(ge=0, le=10)
    note: str | None = Field(default=None, max_length=1000)


class HumanLabelResponse(BaseModel):
    id: int | None
    run_id: int
    question_key: str
    metric: str
    score: float
    labeler_id: int
    note: str | None = None
    update_date: datetime


class HumanLabelsResponse(BaseModel):
    labels: list[HumanLabelResponse]


class JudgeAgreementResponse(BaseModel):
    run_id: int
    dataset: str
    judge_model: str | None = None
    judge_rubric_version: str | None = None
    # Per metric: n, mae, spearman, kappa and min_kappa.
    metrics: dict[str, dict[str, Any]]
