"""Why a benchmark question failed: one class per result, computed from stored metrics only.

Pure rules over a persisted per-question result, so old runs are classified again
without repeating paid generation or judging.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from domain.benchmark.question import ABSTENTION_TYPES, answer_type
from domain.rag.outcome import AnswerOutcome


class DiagnosisClass(StrEnum):
    RETRIEVAL_MISS = "retrieval_miss"
    CONTEXT_LOSS = "context_loss"
    GENERATION_ERROR = "generation_error"
    FALSE_REFUSAL = "false_refusal"
    FALSE_ANSWER = "false_answer"
    OK = "ok"
    UNLABELED = "unlabeled"


@dataclass(frozen=True)
class DiagnosisThresholds:
    min_generation_score: float = 7.0
    min_refusal_score: float = 7.0


DEFAULT_DIAGNOSIS_THRESHOLDS = DiagnosisThresholds()

# Completion strings written before AnswerOutcome existed.
LEGACY_COMPLETIONS = {
    "grounded_response": AnswerOutcome.GROUNDED,
    "input_safety": AnswerOutcome.INPUT_BLOCKED,
    "output_safety": AnswerOutcome.OUTPUT_BLOCKED,
    "relevance_refusal": AnswerOutcome.NOT_FOUND,
    "sufficiency_refusal": AnswerOutcome.NOT_FOUND,
    "prompt_budget": AnswerOutcome.ERROR,
}

REFUSAL_OUTCOMES = frozenset(
    {
        AnswerOutcome.NOT_FOUND,
        AnswerOutcome.OUT_OF_DOMAIN,
        AnswerOutcome.INPUT_BLOCKED,
        AnswerOutcome.OUTPUT_BLOCKED,
    }
)
ANSWERED_OUTCOMES = frozenset({AnswerOutcome.GENERATED, AnswerOutcome.GROUNDED, AnswerOutcome.CACHE_HIT})


def result_outcome(result: dict[str, Any]) -> AnswerOutcome | None:
    """Stored outcome; older runs only kept the trace completion string."""
    value = result.get("outcome") or (result.get("evidence") or {}).get("completion")

    if not isinstance(value, str):
        return None

    if value in LEGACY_COMPLETIONS:
        return LEGACY_COMPLETIONS[value]

    try:
        return AnswerOutcome(value)
    except ValueError:
        return None


def metric(result: dict[str, Any], group: str, name: str) -> float | None:
    value = (result.get(group) or {}).get(name)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def diagnose_abstention(
    result: dict[str, Any], outcome: AnswerOutcome | None, thresholds: DiagnosisThresholds
) -> DiagnosisClass:
    if outcome in REFUSAL_OUTCOMES:
        return DiagnosisClass.OK

    if outcome not in ANSWERED_OUTCOMES:
        return DiagnosisClass.UNLABELED

    # A generated text may still be an honest refusal; the judge's behaviour score decides.
    refusal = metric(result, "evidence_metrics", "refusal_score")
    honest = refusal is not None and refusal >= thresholds.min_refusal_score
    return DiagnosisClass.OK if honest else DiagnosisClass.FALSE_ANSWER


def retrieval_recall(result: dict[str, Any]) -> float | None:
    recall = metric(result, "evidence_metrics", "fragment_recall_at_k")

    if recall is not None:
        return recall

    return metric(result, "evidence_metrics", "retrieval_fact_coverage")


def has_evidence_labels(result: dict[str, Any]) -> bool:
    annotations = result.get("annotations") or {}
    return bool(annotations.get("expected_fragments") or annotations.get("required_facts"))


def answer_failure(
    result: dict[str, Any], outcome: AnswerOutcome | None, thresholds: DiagnosisThresholds
) -> DiagnosisClass | None:
    """First failing pipeline stage of an answerable question, or None when every stage passed."""
    retrieved = retrieval_recall(result)
    in_context = metric(result, "evidence_metrics", "context_fragment_recall")
    fragment_recall = metric(result, "evidence_metrics", "fragment_recall_at_k")
    scores = [
        score
        for name in ("correctness", "faithfulness")
        if (score := metric(result, "generator_metrics", name)) is not None
    ]
    # A refusal before retrieval (out of domain, blocked input) has no recall; labels show it was answerable.
    refused = outcome in REFUSAL_OUTCOMES and (retrieved is not None or has_evidence_labels(result))
    rules = (
        (retrieved is not None and retrieved < 1, DiagnosisClass.RETRIEVAL_MISS),
        (refused, DiagnosisClass.FALSE_REFUSAL),
        (
            fragment_recall is not None and in_context is not None and in_context < fragment_recall,
            DiagnosisClass.CONTEXT_LOSS,
        ),
        (any(score < thresholds.min_generation_score for score in scores), DiagnosisClass.GENERATION_ERROR),
        (retrieved is None and not scores, DiagnosisClass.UNLABELED),
    )
    return next((diagnosis for failed, diagnosis in rules if failed), None)


def diagnose(
    result: dict[str, Any], thresholds: DiagnosisThresholds = DEFAULT_DIAGNOSIS_THRESHOLDS
) -> DiagnosisClass:
    """Classify one result; the first failing pipeline stage is the root cause."""
    if (result.get("annotation_validation") or {}).get("status") == "invalid_annotation":
        return DiagnosisClass.UNLABELED

    outcome = result_outcome(result)

    if (result.get("answer_type") or answer_type(result.get("annotations"))) in ABSTENTION_TYPES:
        return diagnose_abstention(result, outcome, thresholds)

    return answer_failure(result, outcome, thresholds) or DiagnosisClass.OK


def with_diagnosis(
    results: list[dict[str, Any]], thresholds: DiagnosisThresholds = DEFAULT_DIAGNOSIS_THRESHOLDS
) -> list[dict[str, Any]]:
    """Copies with `outcome` and a freshly computed `diagnosis`."""
    annotated = []

    for result in results:
        outcome = result_outcome(result)
        annotated.append(
            {
                **result,
                "outcome": outcome.value if outcome is not None else None,
                "diagnosis": diagnose(result, thresholds).value,
            }
        )

    return annotated


def diagnosis_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys((item.value for item in DiagnosisClass), 0)

    for result in results:
        counts[diagnose(result).value] += 1

    return counts


def false_answer_rate(results: list[dict[str, Any]]) -> float | None:
    """Share of expected-abstention questions that received a substantive answer."""
    abstentions = [
        result
        for result in results
        if (result.get("answer_type") or answer_type(result.get("annotations"))) in ABSTENTION_TYPES
    ]

    if not abstentions:
        return None

    wrong = sum(diagnose(result) == DiagnosisClass.FALSE_ANSWER for result in abstentions)
    return round(wrong / len(abstentions), 4)


def result_key(result: dict[str, Any]) -> str:
    """Identity of one evaluated case across runs of the same dataset."""
    identity = result.get("id")

    if identity is None or identity == "":
        identity = result.get("case_id") or result.get("question")

    return f"{identity}#{result.get('run', 1)}"


def diagnosis_transitions(baseline: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, Any]:
    """Class changes per case present in both runs: a from→to matrix and the changed cases."""
    before = {result_key(result): result for result in baseline}
    matrix: dict[str, dict[str, int]] = {}
    changes = []

    for result in current:
        key = result_key(result)
        previous = before.get(key)

        if previous is None:
            continue

        source = diagnose(previous).value
        target = diagnose(result).value
        row = matrix.setdefault(source, {})
        row[target] = row.get(target, 0) + 1

        if source != target:
            changes.append(
                {
                    "key": key,
                    "id": result.get("id"),
                    "question": result.get("question"),
                    "from": source,
                    "to": target,
                }
            )

    return {
        "matrix": matrix,
        "changes": changes,
        "compared": sum(sum(row.values()) for row in matrix.values()),
    }
