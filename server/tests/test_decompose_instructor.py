"""Tests for decompose_question with instructor + DecompositionCheck.

Verifies that:
1. Instructor path returns structured DecompositionCheck results correctly
2. needs_decomposition=False returns [question]
3. Empty sub_queries returns [question]
4. Legacy path (no instructor_client) still works via LangChain
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.ml.clients.llm_schemas import DecompositionCheck  # noqa: E402
from infrastructure.ml.rag.rag_prompts import decompose_question  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_instructor_client(check: DecompositionCheck):
    """Create a mock async instructor client returning the given DecompositionCheck."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=check)
    return client


def _mock_ml_clients(fast_llm_response: str = ""):
    """Create a mock ml_clients with fast_llm and auxiliary_semaphore."""
    from langchain_core.language_models import FakeListChatModel

    llm = FakeListChatModel(responses=[fast_llm_response])
    return SimpleNamespace(
        fast_llm=lambda: llm,
        auxiliary_semaphore=AsyncMock(),
    )


# ---------------------------------------------------------------------------
# Instructor path tests
# ---------------------------------------------------------------------------


class TestDecomposeInstructorPath:
    @pytest.mark.asyncio
    async def test_returns_sub_queries(self):
        check = DecompositionCheck(
            needs_decomposition=True,
            sub_queries=["What is RAG?", "How does retrieval work?"],
        )
        client = _mock_instructor_client(check)
        result = await decompose_question("Tell me about RAG and retrieval", instructor_client=client)
        assert result == ["What is RAG?", "How does retrieval work?"]

    @pytest.mark.asyncio
    async def test_needs_decomposition_false_returns_original(self):
        check = DecompositionCheck(needs_decomposition=False, sub_queries=[])
        client = _mock_instructor_client(check)
        result = await decompose_question("Simple question", instructor_client=client)
        assert result == ["Simple question"]

    @pytest.mark.asyncio
    async def test_empty_sub_queries_returns_original(self):
        check = DecompositionCheck(needs_decomposition=True, sub_queries=[])
        client = _mock_instructor_client(check)
        result = await decompose_question("question", instructor_client=client)
        assert result == ["question"]

    @pytest.mark.asyncio
    async def test_single_sub_query_returns_original(self):
        check = DecompositionCheck(needs_decomposition=True, sub_queries=["only one"])
        client = _mock_instructor_client(check)
        result = await decompose_question("question", instructor_client=client)
        assert result == ["question"]

    @pytest.mark.asyncio
    async def test_strips_empty_strings(self):
        check = DecompositionCheck(
            needs_decomposition=True,
            sub_queries=["q1", "", "  ", "q2"],
        )
        client = _mock_instructor_client(check)
        result = await decompose_question("complex", instructor_client=client)
        assert result == ["q1", "q2"]

    @pytest.mark.asyncio
    async def test_limits_to_four(self):
        check = DecompositionCheck(
            needs_decomposition=True,
            sub_queries=["q1", "q2", "q3", "q4", "q5"],
        )
        client = _mock_instructor_client(check)
        result = await decompose_question("complex", instructor_client=client)
        assert len(result) == 4
        assert result == ["q1", "q2", "q3", "q4"]

    @pytest.mark.asyncio
    async def test_instructor_uses_max_retries_two(self):
        check = DecompositionCheck(needs_decomposition=False, sub_queries=[])
        client = _mock_instructor_client(check)
        await decompose_question("q", instructor_client=client)
        call_kwargs = client.chat.completions.create.call_args
        # Check max_retries=2 in either kwargs or positional args
        assert call_kwargs.kwargs.get("max_retries") == 2 or call_kwargs[1].get("max_retries") == 2

    @pytest.mark.asyncio
    async def test_semaphore_acquired_when_ml_clients_provided(self):
        check = DecompositionCheck(needs_decomposition=False, sub_queries=[])
        client = _mock_instructor_client(check)
        ml_clients = _mock_ml_clients()
        await decompose_question("q", instructor_client=client, ml_clients=ml_clients)
        # TimeoutSemaphore.__aenter__ calls acquire() internally
        ml_clients.auxiliary_semaphore.__aenter__.assert_called_once()

    @pytest.mark.asyncio
    async def test_semaphore_not_acquired_without_ml_clients(self):
        check = DecompositionCheck(needs_decomposition=False, sub_queries=[])
        client = _mock_instructor_client(check)
        # Should not raise — no semaphore needed
        await decompose_question("q", instructor_client=client)


# ---------------------------------------------------------------------------
# Legacy path tests (no instructor_client) -- returns original question
# ---------------------------------------------------------------------------


class TestDecomposeLegacyPath:
    @pytest.mark.asyncio
    async def test_legacy_ml_clients_only_returns_original(self):
        """Without instructor_client, decompose_question returns the original question."""
        ml_clients = _mock_ml_clients(fast_llm_response="q1\nq2\nq3")
        result = await decompose_question("complex question", ml_clients=ml_clients)
        assert result == ["complex question"]

    @pytest.mark.asyncio
    async def test_legacy_single_line_returns_original(self):
        ml_clients = _mock_ml_clients(fast_llm_response="only one question")
        result = await decompose_question("simple question", ml_clients=ml_clients)
        assert result == ["simple question"]

    @pytest.mark.asyncio
    async def test_no_client_no_ml_clients_returns_original(self):
        result = await decompose_question("question")
        assert result == ["question"]


# ---------------------------------------------------------------------------
# Caller integration (helpers.py)
# ---------------------------------------------------------------------------


class TestDecomposeCallerIntegration:
    @pytest.mark.asyncio
    async def test_helpers_passes_instructor_client(self):
        """Verify that helpers.py passes instructor_client to decompose_question."""
        from infrastructure.ml.rag.helpers import retrieve_with_decomposition

        mock_rag = MagicMock()
        mock_rag.features.decomposition_enabled = True
        mock_rag.retriever.fetch_k = 10

        sub_queries = ["q1", "q2"]
        with (
            patch(
                "infrastructure.ml.rag.helpers.decompose_question",
                AsyncMock(return_value=sub_queries),
            ) as mock_decompose,
            patch("infrastructure.ml.rag.helpers.run_hybrid_search", AsyncMock(return_value=[])),
        ):
            mock_ml = MagicMock()
            mock_ml.instructor_client = MagicMock()
            from domain.value_objects.llm_provider import Breadth

            await retrieve_with_decomposition(
                "complex question",
                10,
                None,
                mock_rag,
                mock_ml,
                Breadth.NARROW,
                "general",
                0.7,
                0.3,
                visibility_conditions=[],
                user_id=0,
                user_group_ids=[],
            )
            mock_decompose.assert_called_once()
            call_kwargs = mock_decompose.call_args
            assert call_kwargs.kwargs.get("instructor_client") is mock_ml.instructor_client
