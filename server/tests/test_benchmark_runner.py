"""Characterization tests for infrastructure.benchmark.runner.

Tests database case validation and retriever metrics with real computation logic.
"""

from __future__ import annotations

import pytest

from infrastructure.benchmark.runner import validate_questions


def test_validate_questions_accepts_database_cases():
    validate_questions([{"id": 41, "question": "Q", "annotations": {"expected_refusal": True}}])


@pytest.mark.parametrize("questions", [[], "questions.json", {}, [None], [{"question": 1}]])
def test_validate_questions_rejects_invalid_cases(questions):
    with pytest.raises(ValueError):
        validate_questions(questions)


# ---------------------------------------------------------------------------
# compute_retriever_metrics_from_sources
# ---------------------------------------------------------------------------


def test_compute_retriever_metrics_no_source_hint():
    from infrastructure.benchmark.runner import compute_retriever_metrics_from_sources

    sources = [
        {"source": "doc1.pdf", "max_score": 0.9},
        {"source": "doc2.pdf", "max_score": 0.7},
    ]

    result = compute_retriever_metrics_from_sources(sources, source_hint=None)

    assert result["hit_rate"] is None
    assert result["mrr"] is None
    assert result["avg_similarity"] == 0.8
    assert result["retrieved_sources"] == ["doc1.pdf", "doc2.pdf"]


def test_compute_retriever_metrics_with_hint_hit():
    from infrastructure.benchmark.runner import compute_retriever_metrics_from_sources

    sources = [
        {"source": "contracts/report.pdf", "max_score": 0.9},
        {"source": "other/file.pdf", "max_score": 0.7},
    ]

    result = compute_retriever_metrics_from_sources(sources, source_hint="report")

    assert result["hit_rate"] == 1
    assert result["mrr"] == 1.0  # first position


def test_compute_retriever_metrics_with_hint_hit_at_rank2():
    from infrastructure.benchmark.runner import compute_retriever_metrics_from_sources

    sources = [
        {"source": "other.pdf", "max_score": 0.9},
        {"source": "contracts/report.pdf", "max_score": 0.7},
    ]

    result = compute_retriever_metrics_from_sources(sources, source_hint="report")

    assert result["hit_rate"] == 1
    assert result["mrr"] == 0.5  # 1/2


def test_compute_retriever_metrics_with_hint_no_hit():
    from infrastructure.benchmark.runner import compute_retriever_metrics_from_sources

    sources = [
        {"source": "other.pdf", "max_score": 0.9},
    ]

    result = compute_retriever_metrics_from_sources(sources, source_hint="report")

    assert result["hit_rate"] == 0
    assert result["mrr"] == 0.0


def test_compute_retriever_metrics_empty_sources():
    from infrastructure.benchmark.runner import compute_retriever_metrics_from_sources

    result = compute_retriever_metrics_from_sources([], source_hint="anything")

    assert result["avg_similarity"] == 0.0
    assert result["retrieved_sources"] == []


def test_compute_retriever_metrics_missing_max_score():
    from infrastructure.benchmark.runner import compute_retriever_metrics_from_sources

    sources = [{"source": "doc.pdf"}]  # no max_score key

    result = compute_retriever_metrics_from_sources(sources, source_hint=None)

    assert result["avg_similarity"] == 0.0
