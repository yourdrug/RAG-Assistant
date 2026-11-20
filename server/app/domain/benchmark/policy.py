"""Acceptance rules of one benchmark dataset: absolute floors, measured noise and judge calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
import math
from typing import Any

from domain.shared.exceptions import ValidationError


# Summary metrics compared with a baseline; the regression check and noise share this list.
COMPARED_METRICS = (
    "hit_rate",
    "avg_fragment_recall_at_k",
    "avg_context_fragment_recall",
    "faithfulness",
    "relevancy",
    "correctness",
    "avg_context_precision",
    "avg_context_recall",
)

# Allowed drop against the baseline when no valid noise estimate exists.
DEFAULT_ALLOWED_DROPS = {
    "hit_rate": -0.05,
    "avg_fragment_recall_at_k": -0.05,
    "avg_context_fragment_recall": -0.05,
    "faithfulness": -1.0,
    "relevancy": -1.0,
    "correctness": -1.0,
    "avg_context_precision": -1.0,
    "avg_context_recall": -1.0,
}
# Per-question score range of each compared metric, the summary field counting its scored cases
# and the rounding step of the stored summary value.
QUESTION_SCALES = {
    "hit_rate": ("hit_rate_evaluated_count", 1.0, 0.0001),
    "avg_fragment_recall_at_k": ("fragment_recall_at_k_evaluated_count", 1.0, 0.0001),
    "avg_context_fragment_recall": ("context_fragment_recall_evaluated_count", 1.0, 0.0001),
    "faithfulness": ("faithfulness_evaluated_count", 10.0, 0.1),
    "relevancy": ("relevancy_evaluated_count", 10.0, 0.1),
    "correctness": ("correctness_evaluated_count", 10.0, 0.1),
    "avg_context_precision": ("context_precision_evaluated_count", 10.0, 0.1),
    "avg_context_recall": ("context_recall_evaluated_count", 10.0, 0.1),
}
# Acceptance floor or ceiling → the summary metric it applies to.
ACCEPTANCE_METRICS = {
    "min_fragment_recall_at_k": "avg_fragment_recall_at_k",
    "min_context_fragment_recall": "avg_context_fragment_recall",
    "min_correctness": "correctness",
    "max_false_answer_rate": "false_answer_rate",
}

# A change is significant when it exceeds this many standard deviations of repeated runs.
NOISE_SIGMAS = 2.0
MIN_NOISE_RUNS = 3
MAX_DATASET_LENGTH = 100


class GateStatus(StrEnum):
    """Verdict of a quality gate; missing evidence is never reported as a pass."""

    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class AcceptanceThresholds:
    min_fragment_recall_at_k: float | None = None
    min_context_fragment_recall: float | None = None
    min_correctness: float | None = None
    max_false_answer_rate: float | None = None

    def __post_init__(self) -> None:
        for name in ("min_fragment_recall_at_k", "min_context_fragment_recall", "max_false_answer_rate"):
            bounded(name, getattr(self, name), 1.0)

        bounded("min_correctness", self.min_correctness, 10.0)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "min_fragment_recall_at_k": self.min_fragment_recall_at_k,
            "min_context_fragment_recall": self.min_context_fragment_recall,
            "min_correctness": self.min_correctness,
            "max_false_answer_rate": self.max_false_answer_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AcceptanceThresholds:
        data = data or {}
        return cls(**{name: data.get(name) for name in cls().to_dict()})


def bounded(name: str, value: object, upper: float) -> None:
    if value is None:
        return

    if not isinstance(value, int | float) or isinstance(value, bool) or not 0 <= value <= upper:
        message = f"{name} must be a number between 0 and {upper:g}"
        raise ValidationError(message)


@dataclass(frozen=True)
class JudgeIdentity:
    """Scores are comparable only under the same judge model and rubric."""

    model: str | None
    rubric_version: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {"judge_model": self.model, "judge_rubric_version": self.rubric_version}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeIdentity:
        return cls(model=data.get("judge_model"), rubric_version=data.get("judge_rubric_version"))


def run_judge(config: dict[str, Any], results: list[dict[str, Any]] | None) -> JudgeIdentity:
    """The judge of a run: its configured model and the rubric version stored with each result."""
    rubric = next(
        (row.get("judge_rubric_version") for row in results or [] if row.get("judge_rubric_version")), None
    )
    return JudgeIdentity(model=config.get("judge_model"), rubric_version=rubric)


@dataclass(frozen=True)
class NoiseEstimate:
    sigma: dict[str, float]
    judge: JudgeIdentity
    run_ids: list[int]
    measured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def valid_for(self, judge: JudgeIdentity) -> bool:
        return self.judge == judge

    def to_dict(self) -> dict[str, Any]:
        return {
            "sigma": self.sigma,
            **self.judge.to_dict(),
            "run_ids": self.run_ids,
            "measured_at": self.measured_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NoiseEstimate | None:
        if not data:
            return None

        return cls(
            sigma={key: float(value) for key, value in (data.get("sigma") or {}).items()},
            judge=JudgeIdentity.from_dict(data),
            run_ids=list(data.get("run_ids") or []),
            measured_at=datetime.fromisoformat(data["measured_at"]),
        )


@dataclass(frozen=True)
class JudgeCalibration:
    """Agreement of the judge with human labels for one run."""

    run_id: int | None
    judge: JudgeIdentity
    metrics: dict[str, dict[str, float | int | None]]
    measured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: int | None = None
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            **self.judge.to_dict(),
            "metrics": self.metrics,
            "measured_at": self.measured_at.isoformat(),
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> JudgeCalibration | None:
        if not data:
            return None

        return cls(
            run_id=int(data["run_id"]) if data.get("run_id") is not None else None,
            judge=JudgeIdentity.from_dict(data),
            metrics=dict(data.get("metrics") or {}),
            measured_at=datetime.fromisoformat(data["measured_at"]),
            created_by=data.get("created_by"),
        )


@dataclass
class DatasetPolicy:
    dataset: str
    thresholds: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)
    noise: NoiseEstimate | None = None
    calibration: JudgeCalibration | None = None
    updated_by: int | None = None
    update_date: datetime | None = None

    def __post_init__(self) -> None:
        if not self.dataset.strip() or len(self.dataset) > MAX_DATASET_LENGTH:
            message = "dataset must be a nonempty name of at most 100 characters"
            raise ValidationError(message)


def sample_sigma(values: list[float]) -> float | None:
    if len(values) < 2:  # noqa: PLR2004 — sample deviation needs two values.
        return None

    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def metric_noise(summaries: list[dict[str, Any]]) -> dict[str, float]:
    """Standard deviation of each compared metric over repeated runs of one configuration."""
    noise = {}

    for name in COMPARED_METRICS:
        values = [float(summary[name]) for summary in summaries if summary.get(name) is not None]
        sigma = sample_sigma(values) if len(values) == len(summaries) else None

        if sigma is not None:
            noise[name] = round(sigma, 4)

    return noise


def question_weights(summary: dict[str, Any]) -> dict[str, float]:
    """Shift of each compared metric when a single scored question moves across its whole range.

    One rounding step of the stored summary is added: rounded averages can differ by more than the
    exact shift, and a delta of exactly one question must not fail on float error.
    """
    weights = {}

    for name, (count_key, scale, step) in QUESTION_SCALES.items():
        count = summary.get(count_key)

        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            weights[name] = round(scale / count + step, 4)

    return weights


def allowed_drops(
    noise: NoiseEstimate | None, judge: JudgeIdentity, weights: dict[str, float] | None = None
) -> tuple[dict[str, float], list[str]]:
    """Per-metric allowed drop: 2 sigma of a noise estimate made with the same judge, else the defaults.

    A drop never goes below one question's weight: a few repeated runs can measure sigma as zero,
    and a single changed answer must not fail the gate by itself.
    """
    drops = dict(DEFAULT_ALLOWED_DROPS)
    warnings = []

    if noise is None:
        warnings.append("noise not measured for this dataset; default thresholds apply")
    elif not noise.valid_for(judge):
        warnings.append(
            f"noise was measured with judge {noise.judge.model} (rubric {noise.judge.rubric_version}), "
            f"this run uses {judge.model} (rubric {judge.rubric_version}); default thresholds apply"
        )
    else:
        drops.update(
            {name: -round(NOISE_SIGMAS * sigma, 4) for name, sigma in noise.sigma.items() if name in drops}
        )

    for name, weight in (weights or {}).items():
        if name in drops:
            drops[name] = min(drops[name], -weight)

    return drops, warnings


def acceptance_checks(summary: dict[str, Any], thresholds: AcceptanceThresholds) -> list[dict[str, Any]]:
    """Absolute floors and ceilings of a dataset, checked on the whole current run."""
    checks = []

    for name, limit in thresholds.to_dict().items():
        if limit is None:
            continue

        value = summary.get(ACCEPTANCE_METRICS[name])
        check: dict[str, Any] = {
            "metric": name,
            "kind": "acceptance",
            "baseline": None,
            "current": value,
            "delta": None,
            "threshold": limit,
            "failed": False,
        }

        if value is None:
            # A configured threshold that cannot be checked leaves the gate without evidence.
            check["note"] = "missing data; threshold not checked"
            check["insufficient"] = True
        else:
            check["failed"] = value > limit if name.startswith("max_") else value < limit

        checks.append(check)

    return checks
