"""Human labels of judged answers and the judge's agreement with them."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from domain.benchmark.diagnosis import result_key
from domain.shared.exceptions import ValidationError


MAX_NOTE_LENGTH = 1000
MAX_SCORE = 10.0
# Scores at or above this value count as an acceptable answer when binarised.
PASS_SCORE = 7.0
# Below this Cohen's kappa the judge, not the pipeline, must be fixed first.
MIN_KAPPA = 0.6
MAX_KEY_LENGTH = 200


class LabelMetric(StrEnum):
    CORRECTNESS = "correctness"
    FAITHFULNESS = "faithfulness"


@dataclass
class HumanLabel:
    """One person's score of one judged metric of one case in a run."""

    run_id: int
    question_key: str
    metric: LabelMetric
    score: float
    labeler_id: int
    note: str | None = None
    id: int | None = None
    update_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.metric = LabelMetric(self.metric)

        if not self.question_key or len(self.question_key) > MAX_KEY_LENGTH:
            message = "question_key must be a nonempty case key"
            raise ValidationError(message)

        if isinstance(self.score, bool) or not 0 <= self.score <= MAX_SCORE:
            message = "score must be between 0 and 10"
            raise ValidationError(message)

        if self.note is not None:
            self.note = self.note.strip() or None

        if self.note is not None and len(self.note) > MAX_NOTE_LENGTH:
            message = f"note must be at most {MAX_NOTE_LENGTH} characters"
            raise ValidationError(message)


def ranks(values: list[float]) -> list[float]:
    """Average ranks; ties share the mean of their positions."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0

    while start < len(order):
        end = start

        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1

        for position in range(start, end + 1):
            result[order[position]] = (start + end) / 2 + 1

        start = end + 1

    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    spread = (sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)) ** 0.5
    return covariance / spread if spread else None


def cohen_kappa(left: list[bool], right: list[bool]) -> float | None:
    total = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / total
    yes = (sum(left) / total) * (sum(right) / total)
    no = (1 - sum(left) / total) * (1 - sum(right) / total)
    expected = yes + no

    if expected == 1:
        return None

    return (observed - expected) / (1 - expected)


def agreement(pairs: list[tuple[float, float]]) -> dict[str, float | int | None]:
    """Judge-vs-human agreement over (judge, human) score pairs of one metric."""
    count = len(pairs)

    if count == 0:
        return {"n": 0, "mae": None, "spearman": None, "kappa": None}

    judge = [float(pair[0]) for pair in pairs]
    human = [float(pair[1]) for pair in pairs]
    mae = sum(abs(a - b) for a, b in zip(judge, human, strict=True)) / count
    spearman = pearson(ranks(judge), ranks(human)) if count > 1 else None
    kappa = cohen_kappa([score >= PASS_SCORE for score in judge], [score >= PASS_SCORE for score in human])
    return {
        "n": count,
        "mae": round(mae, 3),
        "spearman": round(spearman, 3) if spearman is not None else None,
        "kappa": round(kappa, 3) if kappa is not None else None,
    }


def judge_agreement(results: list[dict[str, Any]], labels: list[HumanLabel]) -> dict[str, dict[str, Any]]:
    """Per-metric agreement; several labelers of one case are averaged into one human score."""
    judged = {result_key(result): result.get("generator_metrics") or {} for result in results}
    report = {}

    for metric in LabelMetric:
        human: dict[str, list[float]] = {}

        for label in labels:
            if label.metric == metric and label.question_key in judged:
                human.setdefault(label.question_key, []).append(label.score)

        pairs = [
            (judged[key][metric.value], sum(scores) / len(scores))
            for key, scores in human.items()
            if judged[key].get(metric.value) is not None
        ]
        report[metric.value] = {**agreement(pairs), "min_kappa": MIN_KAPPA}

    return report
