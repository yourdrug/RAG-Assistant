"""Benchmark failure classes: one rule per condition, legacy runs and run-to-run transitions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from application.benchmark.diagnosis import compare_diagnoses, rediagnose
from application.benchmark.orchestrator import compute_summary_from_results
from domain.benchmark.diagnosis import (
    DiagnosisClass,
    diagnose,
    diagnosis_counts,
    diagnosis_transitions,
    false_answer_rate,
    result_outcome,
)
from domain.benchmark.question import content_hash
from domain.benchmark.run import BenchmarkRun
from domain.rag.outcome import AnswerOutcome


FRAGMENT = {"source": "a.docx", "text": "ставка 10 процентов"}


def row(  # noqa: PLR0913 — one keyword per stored metric keeps the cases readable
    *,
    outcome: str | None = "generated",
    recall: float | None = 1.0,
    context: float | None = 1.0,
    correctness: float | None = 9.0,
    faithfulness: float | None = 9.0,
    annotations: dict[str, Any] | None = None,
    refusal: float | None = None,
    legacy: bool = False,
    question_id: int = 1,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": question_id,
        "question": f"q{question_id}",
        "run": 1,
        "annotations": {"expected_fragments": [FRAGMENT]} if annotations is None else annotations,
        "evidence_metrics": {
            "fragment_recall_at_k": recall,
            "context_fragment_recall": context,
            "refusal_score": refusal,
        },
        "generator_metrics": {"correctness": correctness, "faithfulness": faithfulness, "relevancy": 9.0},
        "evidence": {"completion": outcome},
    }

    if not legacy:
        result["outcome"] = outcome

    return result


@pytest.mark.parametrize(
    "result,expected",
    [
        (row(), DiagnosisClass.OK),
        (row(recall=0.5, context=0.5), DiagnosisClass.RETRIEVAL_MISS),
        (row(recall=1.0, context=0.0), DiagnosisClass.CONTEXT_LOSS),
        (row(correctness=5.0), DiagnosisClass.GENERATION_ERROR),
        (row(faithfulness=6.9), DiagnosisClass.GENERATION_ERROR),
        (row(outcome="not_found", context=None, correctness=0.0), DiagnosisClass.FALSE_REFUSAL),
        # Refused before retrieval: no recall, but the labels say the question was answerable.
        (
            row(outcome="out_of_domain", recall=None, context=None, correctness=None, faithfulness=None),
            DiagnosisClass.FALSE_REFUSAL,
        ),
        # The first failing stage is the root cause: a refusal after a retrieval miss is a miss.
        (row(outcome="not_found", recall=0.0, context=None), DiagnosisClass.RETRIEVAL_MISS),
        (
            row(annotations={}, recall=None, context=None, correctness=None, faithfulness=None),
            DiagnosisClass.UNLABELED,
        ),
        (row(annotations={}, recall=None, context=None, correctness=4.0), DiagnosisClass.GENERATION_ERROR),
    ],
)
def test_each_condition_has_its_class(result: dict[str, Any], expected: DiagnosisClass) -> None:
    assert diagnose(result) == expected


@pytest.mark.parametrize(
    "outcome,refusal,expected",
    [
        ("generated", None, DiagnosisClass.FALSE_ANSWER),
        ("generated", 3.0, DiagnosisClass.FALSE_ANSWER),
        # The generated text was itself an honest refusal according to the judge.
        ("generated", 9.0, DiagnosisClass.OK),
        ("not_found", None, DiagnosisClass.OK),
        ("out_of_domain", None, DiagnosisClass.OK),
        ("error", None, DiagnosisClass.UNLABELED),
    ],
)
def test_expected_abstention(outcome: str, refusal: float | None, expected: DiagnosisClass) -> None:
    result = row(outcome=outcome, refusal=refusal, annotations={"answer_type": "not_in_corpus"})

    assert diagnose(result) == expected


def test_invalid_annotations_are_unlabeled() -> None:
    result = {**row(recall=0.0), "annotation_validation": {"status": "invalid_annotation"}}

    assert diagnose(result) == DiagnosisClass.UNLABELED


def test_legacy_completion_strings_map_to_outcomes() -> None:
    assert result_outcome(row(outcome="relevance_refusal", legacy=True)) == AnswerOutcome.NOT_FOUND
    assert result_outcome(row(outcome="grounded_response", legacy=True)) == AnswerOutcome.GROUNDED
    assert result_outcome(row(outcome="input_safety", legacy=True)) == AnswerOutcome.INPUT_BLOCKED
    assert result_outcome(row(outcome="something-else", legacy=True)) is None


def test_legacy_refusal_is_classified_without_rerun() -> None:
    legacy = row(outcome="relevance_refusal", legacy=True, context=None, correctness=0.0)

    assert diagnose(legacy) == DiagnosisClass.FALSE_REFUSAL


def test_counts_and_false_answer_rate() -> None:
    abstain = {"answer_type": "out_of_scope"}
    results = [
        row(),
        row(recall=0.0),
        row(annotations=abstain, outcome="generated"),
        row(annotations=abstain, outcome="not_found"),
    ]

    counts = diagnosis_counts(results)

    assert counts["ok"] == 2
    assert counts["retrieval_miss"] == 1
    assert counts["false_answer"] == 1
    assert sum(counts.values()) == len(results)
    assert false_answer_rate(results) == 0.5
    assert false_answer_rate([row()]) is None


def test_transitions_match_cases_and_list_changes() -> None:
    baseline = [row(question_id=1), row(question_id=2), row(question_id=3)]
    current = [row(question_id=1), row(question_id=2, recall=1.0, context=0.0), row(question_id=4)]

    transitions = diagnosis_transitions(baseline, current)

    assert transitions["compared"] == 2
    assert transitions["matrix"] == {"ok": {"ok": 1, "context_loss": 1}}
    assert transitions["changes"] == [
        {"key": "2#1", "id": 2, "question": "q2", "from": "ok", "to": "context_loss"}
    ]


def test_old_run_is_rediagnosed_from_stored_rows() -> None:
    stored = BenchmarkRun(
        id=7,
        summary_metrics={"avg_correctness": 8.0},
        per_question_results=[row(outcome="relevance_refusal", legacy=True, context=None, correctness=0.0)],
    )

    run = rediagnose(stored)

    assert run.per_question_results is not None
    assert run.per_question_results[0]["diagnosis"] == "false_refusal"
    assert run.per_question_results[0]["outcome"] == "not_found"
    assert run.summary_metrics["diagnosis_counts"]["false_refusal"] == 1
    assert run.summary_metrics["avg_correctness"] == 8.0
    assert stored.per_question_results is not None
    assert "diagnosis" not in stored.per_question_results[0]


def test_compare_runs_from_oldest() -> None:
    now = datetime.now(UTC)
    old = BenchmarkRun(id=1, creation_date=now - timedelta(days=1), per_question_results=[row()])
    new = BenchmarkRun(id=2, creation_date=now, per_question_results=[row(recall=0.0)])

    [transition] = compare_diagnoses([new, old])

    assert (transition["from_run_id"], transition["to_run_id"]) == (1, 2)
    assert transition["matrix"] == {"ok": {"retrieval_miss": 1}}


def test_saved_summary_contains_diagnosis() -> None:
    result = {
        **row(),
        "answer": "Ставка 10 процентов.",
        "latency_sec": 1.0,
        "retriever_metrics": {"hit_rate": 1, "mrr": 1.0, "avg_similarity": 0.0},
        "context_metrics": {},
    }

    summary = compute_summary_from_results([result])

    assert summary["diagnosis_counts"]["ok"] == 1
    assert summary["results"][0]["diagnosis"] == "ok"


def test_content_hash_ignores_bookkeeping_fields() -> None:
    question = {"question": "Q?", "expected_answer": "A", "annotations": {"required_facts": ["A"]}}

    same = content_hash({**question, "tags": ["x"], "notes": "n", "is_active": False, "id": 9})
    changed = content_hash({**question, "expected_answer": "B"})

    assert content_hash(question) == same
    assert content_hash(question) != changed
