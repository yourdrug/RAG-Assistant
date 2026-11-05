"""Tests for judge.py — single-layer retry, client caching, no double retry.

Verifies the fixes for:
1. Double retry (tenacity + instructor) removed — only instructor(max_retries=2) remains
2. get_judge_client caches clients per model (no new TCP pool per call)
3. judge_answer_async._judge_one uses asyncio.wait_for, not manual retry loop
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from infrastructure.benchmark.judging.clients import _judge_client_cache, get_judge_client
from infrastructure.benchmark.judging.judge import (
    JUDGE_OUTPUT_TOKENS,
    judge_answer_async,
    judge_with_structured_output,
)
from infrastructure.budget.runtime import BudgetRuntime


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.ml.clients.llm_schemas import JudgeScore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_instructor_client(return_score: float = 8.0, return_reason: str = "good") -> MagicMock:
    """Create a mock instructor client that returns a valid JudgeScore."""
    client = MagicMock()
    result = JudgeScore(score=return_score, reason=return_reason)
    client.chat.completions.create.return_value = result
    return client


# ---------------------------------------------------------------------------
# judge_with_structured_output — no tenacity, only instructor
# ---------------------------------------------------------------------------


class TestJudgeWithStructuredOutput:
    def test_returns_judge_score(self) -> None:

        client = _mock_instructor_client(return_score=7.5, return_reason="decent")
        result = judge_with_structured_output(client, "test prompt", "model")
        assert isinstance(result, JudgeScore)
        assert result.score == 7.5
        client.chat.completions.create.assert_called_once()

    def test_no_tenacity_retry_on_failure(self) -> None:
        """When instructor raises, judge_with_structured_output should NOT retry externally.

        The old code had @retry(stop=stop_after_attempt(3)) which would catch
        the exception and retry up to 3 times.  Now it should propagate immediately.
        """

        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("LLM unavailable")

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            judge_with_structured_output(client, "prompt", "model")

        # Called exactly once — no outer retry
        assert client.chat.completions.create.call_count == 1

    def test_instructor_max_retries_is_two(self) -> None:
        """Verify the instructor call uses max_retries=2 (not 3 or more)."""

        client = _mock_instructor_client()
        judge_with_structured_output(client, "prompt", "model")

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["max_retries"].stop.max_attempt_number == 2
        assert call_kwargs.kwargs["max_tokens"] == JUDGE_OUTPUT_TOKENS


# ---------------------------------------------------------------------------
# get_judge_client — caching
# ---------------------------------------------------------------------------


class TestGetJudgeClientCaching:
    @pytest.fixture(autouse=True)
    def budget_runtime(self) -> None:
        self.runtime = BudgetRuntime()

    def test_same_model_returns_same_client(self) -> None:

        _judge_client_cache.clear()
        mock_client = MagicMock()
        with patch(
            "infrastructure.benchmark.judging.clients.instructor.from_openai",
            return_value=mock_client,
        ):
            c1 = get_judge_client("test-model", runtime=self.runtime)
            c2 = get_judge_client("test-model", runtime=self.runtime)
            assert c1 is c2
        _judge_client_cache.clear()

    def test_different_models_return_different_clients(self) -> None:

        _judge_client_cache.clear()
        mock_a = MagicMock()
        mock_b = MagicMock()

        with patch(
            "infrastructure.benchmark.judging.clients.instructor.from_openai",
            side_effect=[mock_a, mock_b],
        ):
            c1 = get_judge_client("model-a", runtime=self.runtime)
            c2 = get_judge_client("model-b", runtime=self.runtime)
            assert c1 is not c2
        _judge_client_cache.clear()

    def test_create_called_only_once_per_model(self) -> None:

        _judge_client_cache.clear()
        mock_client = MagicMock()
        with patch(
            "infrastructure.benchmark.judging.clients.instructor.from_openai",
            return_value=mock_client,
        ) as mock_create:
            get_judge_client("x", runtime=self.runtime)
            get_judge_client("x", runtime=self.runtime)
            get_judge_client("x", runtime=self.runtime)
            assert mock_create.call_count == 1
        _judge_client_cache.clear()


# ---------------------------------------------------------------------------
# judge_answer_async._judge_one — asyncio.wait_for, no manual loop
# ---------------------------------------------------------------------------


class TestJudgeAnswerAsync:
    @pytest.mark.asyncio
    async def test_returns_scores(self) -> None:

        client = _mock_instructor_client(return_score=9.0, return_reason="excellent")
        with (
            patch("infrastructure.benchmark.judging.judge.get_judge_model", return_value="test-model"),
            patch("infrastructure.benchmark.judging.judge.get_judge_client", return_value=client),
            patch("infrastructure.benchmark.judging.judge.settings") as mock_settings,
        ):
            mock_settings.llm_provider = SimpleNamespace(value="openrouter")
            mock_settings.llm_auxiliary_timeout = 30.0
            result = await judge_answer_async(
                question="What is RAG?",
                answer="RAG is retrieval-augmented generation",
                context="RAG context",
            )
        assert result["faithfulness"] == 9.0
        assert result["relevancy"] == 9.0

    @pytest.mark.asyncio
    async def test_no_manual_retry_loop(self) -> None:
        """If instructor raises, judge_answer_async should NOT retry manually.

        The old code had a for-attempt loop that retried up to JUDGE_MAX_RETRIES.
        Now each metric's asyncio.gather task propagates immediately on failure.
        """

        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("LLM down")
        with (
            patch("infrastructure.benchmark.judging.judge.get_judge_model", return_value="m"),
            patch("infrastructure.benchmark.judging.judge.get_judge_client", return_value=client),
            patch("infrastructure.benchmark.judging.judge.settings") as mock_settings,
        ):
            mock_settings.llm_provider = SimpleNamespace(value="openrouter")
            mock_settings.llm_auxiliary_timeout = 30.0
            result = await judge_answer_async("q", "a", "ctx")
            assert result["faithfulness"] is None
            assert result["relevancy"] is None
            assert "LLM down" in result["faithfulness_error"]

        # asyncio.gather runs tasks concurrently — when one fails, the others
        # are cancelled.  So we get 1 or 2 calls depending on scheduling, but
        # NOT 3+ (which would indicate a manual retry loop).
        assert client.chat.completions.create.call_count <= 2

    @pytest.mark.asyncio
    async def test_correctness_included_when_expected(self) -> None:

        client = _mock_instructor_client(return_score=7.0, return_reason="ok")
        with (
            patch("infrastructure.benchmark.judging.judge.get_judge_model", return_value="m"),
            patch("infrastructure.benchmark.judging.judge.get_judge_client", return_value=client),
            patch("infrastructure.benchmark.judging.judge.settings") as mock_settings,
        ):
            mock_settings.llm_provider = SimpleNamespace(value="openrouter")
            mock_settings.llm_auxiliary_timeout = 30.0
            result = await judge_answer_async("q", "a", "ctx", expected_answer="expected")
        assert "correctness" in result
        assert result["correctness"] == 7.0
