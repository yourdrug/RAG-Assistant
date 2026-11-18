"""Whether a dataset covers the case types an acceptance set needs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from domain.benchmark.question import ABSTENTION_TYPES, answer_type


MIN_QUESTIONS = 40
MAX_QUESTIONS = 60
# Minimum shares of active questions per case type.
MIN_SHARES = {
    "factual_with_fragments": 0.5,
    "multi_fact": 0.15,
    "abstention": 0.15,
    "with_history": 0.10,
}
# Packs whose documents have editions: answers depend on the date of the question.
DATED_PACKS = frozenset({"belarus_law"})
MIN_DATED_SHARE = 0.10
MULTI_FACT = 2


@dataclass(frozen=True)
class CompositionReport:
    total: int
    counts: dict[str, int]
    shares: dict[str, float]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "counts": self.counts,
            "shares": self.shares,
            "failures": self.failures,
            "passed": self.passed,
        }


def case_types(question: dict[str, Any]) -> set[str]:
    annotations = question.get("annotations") or {}
    kind = answer_type(annotations)
    types = set()

    if kind in ABSTENTION_TYPES:
        types.add("abstention")
    elif kind == "factual" and (annotations.get("expected_fragments") or annotations.get("gold_evidence")):
        types.add("factual_with_fragments")

    if len(annotations.get("required_facts") or []) >= MULTI_FACT:
        types.add("multi_fact")

    if question.get("history"):
        types.add("with_history")

    if question.get("as_of_date"):
        types.add("dated")

    return types


def dataset_composition(questions: list[dict[str, Any]], *, pack_key: str | None = None) -> CompositionReport:
    """Check active questions against the minimum composition of a `-main` acceptance set."""
    active = [question for question in questions if question.get("is_active", True)]
    total = len(active)
    counts = dict.fromkeys((*MIN_SHARES, "dated"), 0)

    for question in active:
        for kind in case_types(question):
            counts[kind] += 1

    shares = {kind: round(count / total, 3) if total else 0.0 for kind, count in counts.items()}
    failures = []

    if not MIN_QUESTIONS <= total <= MAX_QUESTIONS:
        failures.append(f"active questions: {total}, expected {MIN_QUESTIONS}-{MAX_QUESTIONS}")

    minimums = dict(MIN_SHARES)

    if pack_key in DATED_PACKS:
        minimums["dated"] = MIN_DATED_SHARE

    failures.extend(
        f"{kind}: {shares[kind]:.0%} < {minimum:.0%}"
        for kind, minimum in minimums.items()
        if shares[kind] < minimum
    )
    return CompositionReport(total=total, counts=counts, shares=shares, failures=failures)
