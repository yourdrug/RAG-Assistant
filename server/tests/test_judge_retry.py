"""Tests for judge.py — single-layer retry, client caching, no double retry.

Verifies the fixes for:
1. Double retry (tenacity + instructor) removed — only instructor(max_retries=2) remains
2. _get_judge_client caches clients per model (no new TCP pool per call)
3. judge_answer_async._judge_one uses asyncio.wait_for, not manual retry loop
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.ml.clients.llm_schemas import JudgeScore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_instructor_client(return_score: float = 8.0, return_reason: str = "good"):
    """Create a mock instructor client that returns a valid JudgeScore."""
    client = MagicMock()
    result = JudgeScore(score=return_score, reason=return_reason)
    client.chat.completions.create.return_value = result
    return client


# ---------------------------------------------------------------------------
# _judge_with_structured_output — no tenacity, only instructor
# ---------------------------------------------------------------------------


class TestJudgeWithStructuredOutput:
    def test_returns_judge_score(self):
        from infrastructure.benchmark.judge import _judge_with_structured_output

        client = _mock_instructor_client(return_score=7.5, return_reason="decent")
        result = _judge_with_structured_output(client, "test prompt", "model")
        assert isinstance(result, JudgeScore)
        assert result.score == 7.5
        client.chat.completions.create.assert_called_once()

    def test_no_tenacity_retry_on_failure(self):
        """When instructor raises, _judge_with_structured_output should NOT retry externally.

        The old code had @retry(stop=stop_after_attempt(3)) which would catch
        the exception and retry up to 3 times.  Now it should propagate immediately.
        """
        from infrastructure.benchmark.judge import _judge_with_structured_output

        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("LLM unavailable")

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            _judge_with_structured_output(client, "prompt", "model")

        # Called exactly once — no outer retry
        assert client.chat.completions.create.call_count == 1

    def test_instructor_max_retries_is_two(self):
        """Verify the instructor call uses max_retries=2 (not 3 or more)."""
        from infrastructure.benchmark.judge import _judge_with_structured_output

        client = _mock_instructor_client()
        _judge_with_structured_output(client, "prompt", "model")

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("max_retries") == 2


# ---------------------------------------------------------------------------
# _get_judge_client — caching
# ---------------------------------------------------------------------------


class TestGetJudgeClientCaching:
    def test_same_model_returns_same_client(self):
        from infrastructure.benchmark.judge import _get_judge_client, _judge_client_cache

        _judge_client_cache.clear()
        mock_client = MagicMock()
        with patch(
            "infrastructure.ml.clients.instructor_client.create_llm_instructor_client",
            return_value=(mock_client, "resolved_model"),
        ):
            c1 = _get_judge_client("test-model")
            c2 = _get_judge_client("test-model")
            assert c1 is c2
        _judge_client_cache.clear()

    def test_different_models_return_different_clients(self):
        from infrastructure.benchmark.judge import _get_judge_client, _judge_client_cache

        _judge_client_cache.clear()
        mock_a = MagicMock()
        mock_b = MagicMock()

        def _create(model):
            if model == "model-a":
                return mock_a, "a"
            return mock_b, "b"

        with patch(
            "infrastructure.ml.clients.instructor_client.create_llm_instructor_client",
            side_effect=_create,
        ):
            c1 = _get_judge_client("model-a")
            c2 = _get_judge_client("model-b")
            assert c1 is not c2
        _judge_client_cache.clear()

    def test_create_called_only_once_per_model(self):
        from infrastructure.benchmark.judge import _get_judge_client, _judge_client_cache

        _judge_client_cache.clear()
        mock_client = MagicMock()
        with patch(
            "infrastructure.ml.clients.instructor_client.create_llm_instructor_client",
            return_value=(mock_client, "resolved"),
        ) as mock_create:
            _get_judge_client("x")
            _get_judge_client("x")
            _get_judge_client("x")
            assert mock_create.call_count == 1
        _judge_client_cache.clear()


# ---------------------------------------------------------------------------
# judge_answer_async._judge_one — asyncio.wait_for, no manual loop
# ---------------------------------------------------------------------------


class TestJudgeAnswerAsync:
    @pytest.mark.asyncio
    async def test_returns_scores(self):
        from infrastructure.benchmark.judge import judge_answer_async

        client = _mock_instructor_client(return_score=9.0, return_reason="excellent")
        with (
            patch("infrastructure.benchmark.judge._get_judge_model", return_value="test-model"),
            patch("infrastructure.benchmark.judge._get_judge_client", return_value=client),
            patch("infrastructure.benchmark.judge.settings") as mock_settings,
        ):
            mock_settings.llm_provider = SimpleNamespace(value="ollama")
            mock_settings.llm_auxiliary_timeout = 30.0
            result = await judge_answer_async(
                question="What is RAG?",
                answer="RAG is retrieval-augmented generation",
                context="RAG context",
            )
        assert result["faithfulness"] == 9.0
        assert result["relevancy"] == 9.0

    @pytest.mark.asyncio
    async def test_no_manual_retry_loop(self):
        """If instructor raises, judge_answer_async should NOT retry manually.

        The old code had a for-attempt loop that retried up to JUDGE_MAX_RETRIES.
        Now each metric's asyncio.gather task propagates immediately on failure.
        """
        from infrastructure.benchmark.judge import judge_answer_async

        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("LLM down")
        with (
            patch("infrastructure.benchmark.judge._get_judge_model", return_value="m"),
            patch("infrastructure.benchmark.judge._get_judge_client", return_value=client),
            patch("infrastructure.benchmark.judge.settings") as mock_settings,
        ):
            mock_settings.llm_provider = SimpleNamespace(value="ollama")
            mock_settings.llm_auxiliary_timeout = 30.0
            with pytest.raises(RuntimeError, match="LLM down"):
                await judge_answer_async("q", "a", "ctx")

        # asyncio.gather runs tasks concurrently — when one fails, the others
        # are cancelled.  So we get 1 or 2 calls depending on scheduling, but
        # NOT 3+ (which would indicate a manual retry loop).
        assert client.chat.completions.create.call_count <= 2

    @pytest.mark.asyncio
    async def test_correctness_included_when_expected(self):
        from infrastructure.benchmark.judge import judge_answer_async

        client = _mock_instructor_client(return_score=7.0, return_reason="ok")
        with (
            patch("infrastructure.benchmark.judge._get_judge_model", return_value="m"),
            patch("infrastructure.benchmark.judge._get_judge_client", return_value=client),
            patch("infrastructure.benchmark.judge.settings") as mock_settings,
        ):
            mock_settings.llm_provider = SimpleNamespace(value="ollama")
            mock_settings.llm_auxiliary_timeout = 30.0
            result = await judge_answer_async("q", "a", "ctx", expected_answer="expected")
        assert "correctness" in result
        assert result["correctness"] == 7.0


# ---------------------------------------------------------------------------
# get_rag_answer_with_usage — still uses manual retry (LangChain, not instructor)
# ---------------------------------------------------------------------------


class TestGetRagAnswerWithUsage:
    def test_retries_on_failure(self):
        from infrastructure.benchmark.judge import get_rag_answer_with_usage

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "final answer"
        mock_llm.invoke.side_effect = [Exception("transient"), mock_response]

        docs = [(SimpleNamespace(page_content="ctx", metadata={}), 0.9)]
        answer, resp = get_rag_answer_with_usage(mock_llm, docs, "question")
        assert answer == "final answer"
        assert mock_llm.invoke.call_count == 2
