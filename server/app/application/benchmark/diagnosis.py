"""Apply the current diagnosis rule to persisted runs without re-running them."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from domain.benchmark.diagnosis import (
    diagnosis_counts,
    diagnosis_transitions,
    false_answer_rate,
    with_diagnosis,
)


if TYPE_CHECKING:
    from domain.benchmark.run import BenchmarkRun


def rediagnose(run: BenchmarkRun) -> BenchmarkRun:
    """Copy of a fully loaded run whose rows and summary carry freshly computed classes."""
    if not run.per_question_results:
        return run

    rows = with_diagnosis(run.per_question_results)
    summary = {
        **run.summary_metrics,
        "diagnosis_counts": diagnosis_counts(rows),
        "false_answer_rate": false_answer_rate(rows),
    }
    return replace(run, per_question_results=rows, summary_metrics=summary)


def compare_diagnoses(runs: list[BenchmarkRun]) -> list[dict[str, Any]]:
    """Transitions from the oldest run to each newer run, matched by case."""
    ordered = sorted(runs, key=lambda run: run.creation_date)

    if not ordered or not ordered[0].per_question_results:
        return []

    baseline = ordered[0]
    return [
        {
            "from_run_id": baseline.id,
            "to_run_id": run.id,
            **diagnosis_transitions(baseline.per_question_results or [], run.per_question_results or []),
        }
        for run in ordered[1:]
        if run.per_question_results
    ]
