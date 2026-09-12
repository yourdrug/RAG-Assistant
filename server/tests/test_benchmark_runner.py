"""Characterization tests for infrastructure.benchmark.runner.

Tests load_questions and _compute_retriever_metrics_from_sources with
mocked file I/O and real computation logic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from domain.exceptions import BenchmarkQuestionsNotFound


# ---------------------------------------------------------------------------
# load_questions
# ---------------------------------------------------------------------------


def test_load_questions_reads_valid_json(tmp_path):
    from infrastructure.benchmark.runner import load_questions

    questions = [
        {"id": "q1", "question": "What is RAG?", "expected_answer": None, "source_hint": None},
        {
            "id": "q2",
            "question": "How does retrieval work?",
            "expected_answer": "By searching",
            "source_hint": "retrieval",
        },
    ]
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(questions), encoding="utf-8")

    result = load_questions(str(p))

    assert len(result) == 2
    assert result[0]["id"] == "q1"
    assert result[1]["question"] == "How does retrieval work?"


def test_load_questions_creates_example_and_raises(tmp_path):
    from infrastructure.benchmark.runner import EXAMPLE_QUESTIONS, load_questions

    p = tmp_path / "nonexistent.json"

    with pytest.raises(BenchmarkQuestionsNotFound) as exc_info:
        load_questions(str(p))

    assert "nonexistent.json" in str(exc_info.value)

    # Verify example file was created
    assert p.exists()
    created = json.loads(p.read_text(encoding="utf-8"))
    assert created == EXAMPLE_QUESTIONS


def test_load_questions_empty_list(tmp_path):
    from infrastructure.benchmark.runner import load_questions

    p = tmp_path / "empty.json"
    p.write_text("[]", encoding="utf-8")

    result = load_questions(str(p))
    assert result == []


# ---------------------------------------------------------------------------
# _compute_retriever_metrics_from_sources
# ---------------------------------------------------------------------------


def test_compute_retriever_metrics_no_source_hint():
    from infrastructure.benchmark.runner import _compute_retriever_metrics_from_sources

    sources = [
        {"source": "doc1.pdf", "max_score": 0.9},
        {"source": "doc2.pdf", "max_score": 0.7},
    ]

    result = _compute_retriever_metrics_from_sources(sources, source_hint=None)

    assert result["hit_rate"] is None
    assert result["mrr"] is None
    assert result["avg_similarity"] == 0.8
    assert result["retrieved_sources"] == ["doc1.pdf", "doc2.pdf"]


def test_compute_retriever_metrics_with_hint_hit():
    from infrastructure.benchmark.runner import _compute_retriever_metrics_from_sources

    sources = [
        {"source": "contracts/report.pdf", "max_score": 0.9},
        {"source": "other/file.pdf", "max_score": 0.7},
    ]

    result = _compute_retriever_metrics_from_sources(sources, source_hint="report")

    assert result["hit_rate"] == 1
    assert result["mrr"] == 1.0  # first position


def test_compute_retriever_metrics_with_hint_hit_at_rank2():
    from infrastructure.benchmark.runner import _compute_retriever_metrics_from_sources

    sources = [
        {"source": "other.pdf", "max_score": 0.9},
        {"source": "contracts/report.pdf", "max_score": 0.7},
    ]

    result = _compute_retriever_metrics_from_sources(sources, source_hint="report")

    assert result["hit_rate"] == 1
    assert result["mrr"] == 0.5  # 1/2


def test_compute_retriever_metrics_with_hint_no_hit():
    from infrastructure.benchmark.runner import _compute_retriever_metrics_from_sources

    sources = [
        {"source": "other.pdf", "max_score": 0.9},
    ]

    result = _compute_retriever_metrics_from_sources(sources, source_hint="report")

    assert result["hit_rate"] == 0
    assert result["mrr"] == 0.0


def test_compute_retriever_metrics_empty_sources():
    from infrastructure.benchmark.runner import _compute_retriever_metrics_from_sources

    result = _compute_retriever_metrics_from_sources([], source_hint="anything")

    assert result["avg_similarity"] == 0.0
    assert result["retrieved_sources"] == []


def test_compute_retriever_metrics_missing_max_score():
    from infrastructure.benchmark.runner import _compute_retriever_metrics_from_sources

    sources = [{"source": "doc.pdf"}]  # no max_score key

    result = _compute_retriever_metrics_from_sources(sources, source_hint=None)

    assert result["avg_similarity"] == 0.0
