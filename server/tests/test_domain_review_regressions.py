"""Regression cases for date, reading and fragment boundary handling."""

from datetime import date
import re

import pytest

from domain.benchmark.evaluation import matches_fragment
from domain.ingestion.profiles.base import extract_effective_date_generic
from domain.rag.classification import explicit_question_date
from infrastructure.ml.evidence_profiles.belarus.reading import evidence_reading_rules
from infrastructure.ml.rag.rag_prompts import build_prompt


@pytest.mark.parametrize("question", ["цена 12.05.2024", "на 3 этапе 2025"])
def test_non_temporal_questions_do_not_extract_date(question: str) -> None:
    assert explicit_question_date(question) is None


@pytest.mark.parametrize("question", ["на 12.05.2024", "по состоянию на 12 мая 2024"])
def test_explicit_date_is_preserved(question: str) -> None:
    assert explicit_question_date(question) == date(2024, 5, 12)


@pytest.mark.parametrize("question", ["актов", "фактов", "директор", "полезно", "правовой"])
def test_reading_triggers_do_not_match_inside_words(question: str) -> None:
    assert evidence_reading_rules(question) == []


def test_effective_date_skips_unparseable_first_match() -> None:
    result = extract_effective_date_generic(
        "вступает в силу со дня опубликования; вступает в силу 12.05.2024",
        [(re.compile(r"вступает в силу ([^;]+)"), 0.9)],
        re.compile(r"^(never)$"),
    )
    assert result is not None
    assert result.effective_from == date(2024, 5, 12)


def test_fragment_accepts_null_page_end() -> None:
    assert matches_fragment(
        {"source": "a.pdf", "pages": [2], "text": "text"},
        {"metadata": {"source": "a.pdf", "page_start": 2, "page_end": None}, "content": "text"},
    )


def test_addendum_braces_are_literal_after_template_formatting() -> None:
    messages = build_prompt(domain_addendum="Literal {example}").format_messages(
        context="",
        history=[],
        question="test",
    )
    assert "Literal {example}" in messages[0].content
    assert "{{example}}" not in messages[0].content
