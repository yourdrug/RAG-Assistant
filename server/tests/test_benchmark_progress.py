"""Progress reporting accepts unavailable scores and optional judge explanations."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest

from infrastructure.benchmark.evaluation.answer_generators import BenchmarkAnswer
from infrastructure.benchmark.evaluation.case_evaluator import BenchmarkCaseEvaluator
from infrastructure.benchmark.reporting.progress import log_question_result, log_summary


if TYPE_CHECKING:
    from domain.rag.context import RagContext
    from infrastructure.benchmark.evaluation.annotation_validation import AnnotationValidator


@pytest.mark.asyncio
async def test_invalid_annotation_result_can_be_logged(caplog: pytest.LogCaptureFixture) -> None:
    generator = SimpleNamespace(
        generate=AsyncMock(return_value=BenchmarkAnswer("answer", "", {}, None, None))
    )
    validator = SimpleNamespace(
        validate=AsyncMock(return_value={"status": "invalid_annotation", "missing_fragments": [0]})
    )
    question = {"question": "Q"}
    result = await BenchmarkCaseEvaluator(
        generator, "judge", annotation_validator=cast("AnnotationValidator", validator)
    ).evaluate(1, question, 1, cast("RagContext", SimpleNamespace()))

    with caplog.at_level(logging.INFO, logger="default"):
        log_question_result(1, 1, question, result)
        log_summary([result], 1)

    assert result["generator_metrics"]["faithfulness"] is None
    assert "Faithfulness: None/10" in caplog.text
    assert "Ответ: answer" in caplog.text


@pytest.mark.asyncio
async def test_scores_without_reasons_can_be_logged_including_low_quality_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    generator = SimpleNamespace(
        generate=AsyncMock(return_value=BenchmarkAnswer("answer", "", {}, None, None))
    )
    validator = SimpleNamespace(validate=AsyncMock(return_value={"status": "invalid_annotation"}))
    question = {"question": "Q"}
    result = await BenchmarkCaseEvaluator(
        generator, "judge", annotation_validator=cast("AnnotationValidator", validator)
    ).evaluate(1, question, 1, cast("RagContext", SimpleNamespace()))
    result["generator_metrics"] = {"faithfulness": 0, "relevancy": 4, "correctness": 3}

    with caplog.at_level(logging.INFO, logger="default"):
        log_question_result(1, 1, question, result)
        log_summary([result], 1)

    assert "Correctness:  3/10" in caplog.text
    assert "Проблемные вопросы (1)" in caplog.text
