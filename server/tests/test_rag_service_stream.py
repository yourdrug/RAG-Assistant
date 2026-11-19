"""Characterization tests for RagService.stream() — the 270-line god-method.

Locks down the full RAG pipeline orchestration BEFORE refactoring.
Every test mocks external I/O (Qdrant, LLM, Redis) and verifies
the event sequence produced by stream().

"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

from application.rag.stream_events import PipelineMetaEvent, SourcesEvent, StatusEvent, TextChunk
from composition.service_providers import create_rag_service
from domain.identity.user_context import UserContext
from domain.rag.breadth import AnswerStyle, Breadth, output_breadth
from domain.rag.classification import classify_question_breadth
from domain.rag.context import RagContext
from domain.rag.retrieval import compute_retrieval_params
from infrastructure.adapters.pii_redactor_adapter import PIIRedactorAdapter
from infrastructure.ml.rag.generation_budget import prepare_generation_budget
from infrastructure.ml.rag.prompt_blocks import NOT_FOUND_ANSWER
from infrastructure.ml.rag.rag_postprocess import apply_citation_filter
from infrastructure.ml.rag.rag_steps import RagOperations
from tests.test_chat_contracts import chat_fixture
from tests.test_chat_logs import run_attempt
from tests.test_rag_pipeline import make_rag


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from application.ports.ml_clients import LLMMessagePort
    from application.rag.service import RagService
    from application.rag.stream_events import StreamEvent
    from domain.rag.settings import RagSettings
    from infrastructure.ml.rag.generation_budget import GenerationBudget


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(**overrides: Any) -> RagContext:
    defaults: dict[str, Any] = {
        "user_id": 1,
        "user_kind": "internal",
        "user_role": "admin",
        "user_group_ids": [10],
        "depth": None,
        "summary": None,
        "as_of_date": None,
    }
    defaults.update(overrides)
    user = UserContext(
        defaults.pop("user_id"),
        defaults.pop("user_kind"),
        defaults.pop("user_role"),
        tuple(defaults.pop("user_group_ids")),
    ).with_curator_scope(defaults.pop("curator_scope", None))
    return RagContext(user=user, **defaults)


async def _async_iter(items: Sequence[LLMMessagePort]) -> AsyncIterator[LLMMessagePort]:
    for item in items:
        yield item


def make_rag_settings(**feature_overrides: Any) -> RagSettings:

    rag = make_rag()
    features = {
        "condense_enabled": False,
        "decomposition_enabled": False,
        "cache_enabled": False,
        "relevance_gate_enabled": False,
        "citation_filter_enabled": False,
        "rolling_summary_enabled": False,
    }
    features.update(feature_overrides)
    return replace(rag, features=replace(rag.features, **features))


def make_stream_service(llm_response: str = "Ответ из документов.", **overrides: Any) -> RagService:
    mock_ml = MagicMock()
    chunk = MagicMock()
    chunk.content = llm_response
    mock_ml.llm_for_breadth.return_value.astream = MagicMock(return_value=_async_iter([chunk]))
    mock_ml.reranker.return_value.predict_sync = MagicMock(return_value=[0.9, 0.8])
    mock_ml.fast_llm.return_value = MagicMock()
    mock_ml.llm_semaphore = asyncio.Semaphore(1)
    mock_ml.embeddings.return_value.embed_query = AsyncMock(return_value=[0.1] * 384)
    mock_ml.qdrant_client.return_value.search.return_value = []
    mock_ml.qdrant_client.return_value.scroll.return_value = ([], None)
    mock_ml.ensure_bm25_loaded = AsyncMock(return_value=None)
    mock_ml.bm25_search_semaphore = asyncio.Semaphore(8)
    mock_ml.reranker_semaphore = asyncio.Semaphore(8)
    mock_ml.embedding_semaphore = asyncio.Semaphore(8)

    chunk_search = overrides.pop("chunk_search", None)
    domain_keys = overrides.pop("domain_keys", None)
    access = SimpleNamespace(allowed_document_ids=AsyncMock(side_effect=lambda ids, _user: set(ids)))
    return create_rag_service(
        ml_clients=mock_ml, chunk_search=chunk_search, domain_keys=domain_keys, document_access=access
    )


async def collect_events(
    service: RagService, question: str, history: list[object] | None = None, ctx: RagContext | None = None
) -> list[StreamEvent]:
    if history is None:
        history = []

    if ctx is None:
        ctx = _make_context()

    events = []
    async with asyncio.timeout(5):
        events.extend([event async for event in service.stream(question, history, ctx)])
    return events


# ---------------------------------------------------------------------------
# Patch all I/O for stream() tests
# ---------------------------------------------------------------------------

_STREAM_PATCHES = {
    "infrastructure.ml.rag.retrieval_pipeline.run_hybrid_search": AsyncMock(return_value=[]),
    "infrastructure.ml.rag.rag_steps.get_corpus_revision": AsyncMock(return_value="0"),
    "infrastructure.ml.rag.rag_retrieval.ensure_embedding_identity": MagicMock(return_value="test:model:v1"),
    "infrastructure.ml.rag.rag_steps.build_qdrant_filter": MagicMock(return_value=MagicMock(should=[])),
    "infrastructure.ml.rag.rag_steps.build_search_filter": MagicMock(return_value=MagicMock(should=[])),
    "infrastructure.ml.rag.rag_steps.handle_relevance_gate": AsyncMock(return_value=(True, "")),
    "infrastructure.ml.cache.keys.compute_visibility_scope_hash": MagicMock(return_value="vis_hash"),
    "infrastructure.ml.cache.keys.compute_question_hash": MagicMock(return_value="q_hash"),
    "infrastructure.ml.rag.rag_steps.classify_query_domain": MagicMock(return_value="general"),
    "infrastructure.ml.rag.rag_steps.is_out_of_domain": MagicMock(return_value=False),
    "infrastructure.ml.rag.rag_steps.condense_question": AsyncMock(return_value="condensed question"),
    "infrastructure.ml.rag.generation_budget.build_prompt": MagicMock(
        return_value=MagicMock(
            messages=[SystemMessage(content="system prompt")],
            format_messages=lambda **_kw: [
                SystemMessage(content="system prompt"),
                HumanMessage(content="user msg"),
            ],
        )
    ),
    "infrastructure.ml.rag.rag_steps.format_docs_with_selection": MagicMock(
        side_effect=lambda docs, **_kwargs: ("formatted context", list(docs))
    ),
    "infrastructure.ml.rag.rag_steps.extract_sources": MagicMock(
        return_value=[{"source": "a.pdf", "score": 0.9}]
    ),
    "infrastructure.ml.rag.rag_steps.record_rag_answer": MagicMock(),
    "infrastructure.ml.rag.rag_steps.record_llm_usage": MagicMock(),
    "infrastructure.ml.rag.rag_steps.extract_usage_from_langchain": MagicMock(return_value=(100, 50)),
    "infrastructure.ml.rag.rag_steps.request_id_ctx": MagicMock(
        get=MagicMock(return_value="test-req-id"), set=MagicMock()
    ),
    "infrastructure.ml.rag.rag_steps.classify_question_breadth": MagicMock(return_value="narrow"),
    "infrastructure.ml.rag.retrieval_pipeline.should_enumerate_cases": MagicMock(return_value=False),
}


@pytest.fixture(autouse=True)
def _patch_stream_deps() -> Iterator[None]:
    """Patch all I/O-dependent imports for RagService.stream()."""
    # ExitStack undoes already-started patches if a later target is missing,
    # so a failed setup cannot leak mocks into other test modules.
    with ExitStack() as stack:
        for target, mock_val in _STREAM_PATCHES.items():
            stack.enter_context(patch(target, mock_val))

        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamEventSequence:
    @pytest.mark.asyncio
    async def test_yields_status_searching(self) -> None:
        with patch(
            "infrastructure.ml.rag.rag_steps.build_rag_settings",
            return_value=make_rag_settings(),
        ):
            service = make_stream_service(llm_response="test answer")
            events = await collect_events(service, "What is X?")
            stages = [e.stage for e in events if isinstance(e, StatusEvent)]
            assert "searching" in stages

    @pytest.mark.asyncio
    async def test_yields_status_reranking(self) -> None:
        with patch(
            "infrastructure.ml.rag.rag_steps.build_rag_settings",
            return_value=make_rag_settings(),
        ):
            service = make_stream_service()
            events = await collect_events(service, "What is X?")
            stages = [e.stage for e in events if isinstance(e, StatusEvent)]
            assert "reranking" in stages

    @pytest.mark.asyncio
    async def test_yields_status_generating(self) -> None:
        with patch(
            "infrastructure.ml.rag.rag_steps.build_rag_settings",
            return_value=make_rag_settings(),
        ):
            service = make_stream_service()
            events = await collect_events(service, "What is X?")
            stages = [e.stage for e in events if isinstance(e, StatusEvent)]
            assert "generating" in stages

    @pytest.mark.asyncio
    async def test_yields_text_chunk(self) -> None:
        with patch(
            "infrastructure.ml.rag.rag_steps.build_rag_settings",
            return_value=make_rag_settings(),
        ):
            service = make_stream_service(llm_response="Hello world")
            events = await collect_events(service, "question")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert len(text_chunks) >= 1
            assert text_chunks[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_yields_sources_event_last(self) -> None:
        with patch(
            "infrastructure.ml.rag.rag_steps.build_rag_settings",
            return_value=make_rag_settings(),
        ):
            service = make_stream_service()
            events = await collect_events(service, "question")
            assert isinstance(events[-1], PipelineMetaEvent)

    @pytest.mark.asyncio
    async def test_sources_event_has_sources(self) -> None:
        with patch(
            "infrastructure.ml.rag.rag_steps.build_rag_settings",
            return_value=make_rag_settings(),
        ):
            service = make_stream_service()
            events = await collect_events(service, "question")
            sources_event = next(e for e in events if isinstance(e, SourcesEvent))
            assert isinstance(sources_event.sources, list)


class TestStreamOutOfDomain:
    @pytest.mark.asyncio
    async def test_out_of_domain_yields_not_found(self) -> None:
        patch("infrastructure.ml.rag.rag_steps.is_out_of_domain", MagicMock(return_value=True)).start()
        with patch(
            "infrastructure.ml.rag.rag_steps.build_rag_settings",
            return_value=make_rag_settings(),
        ):
            service = make_stream_service()
            events = await collect_events(service, "tell me a joke")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert len(text_chunks) == 1
            assert "не найдена" in text_chunks[0].text.lower()

    @pytest.mark.asyncio
    async def test_out_of_domain_yields_empty_sources(self) -> None:
        patch("infrastructure.ml.rag.rag_steps.is_out_of_domain", MagicMock(return_value=True)).start()
        with patch(
            "infrastructure.ml.rag.rag_steps.build_rag_settings",
            return_value=make_rag_settings(),
        ):
            service = make_stream_service()
            events = await collect_events(service, "joke?")
            sources_event = next(e for e in events if isinstance(e, SourcesEvent))
            assert sources_event.sources == []

    @pytest.mark.asyncio
    async def test_out_of_domain_does_not_call_llm(self) -> None:
        patch("infrastructure.ml.rag.rag_steps.is_out_of_domain", MagicMock(return_value=True)).start()
        with patch(
            "infrastructure.ml.rag.rag_steps.build_rag_settings",
            return_value=make_rag_settings(),
        ):
            service = make_stream_service()
            await collect_events(service, "joke?")
            cast(
                "MagicMock", cast("RagOperations", service.preparation).ml
            ).llm_for_breadth.return_value.astream.assert_not_called()


class TestStreamRelevanceGate:
    @pytest.mark.asyncio
    async def test_rejection_yields_not_relevant_message(self) -> None:
        with (
            patch(
                "infrastructure.ml.rag.rag_steps.build_rag_settings",
                return_value=make_rag_settings(relevance_gate_enabled=True),
            ),
            patch(
                "infrastructure.ml.rag.rag_steps.handle_relevance_gate",
                AsyncMock(return_value=(False, "better query")),
            ),
        ):
            service = make_stream_service()
            events = await collect_events(service, "obscure question")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert len(text_chunks) == 1
            assert text_chunks[0].text == NOT_FOUND_ANSWER

    @pytest.mark.asyncio
    async def test_pass_yields_answer(self) -> None:
        with (
            patch(
                "infrastructure.ml.rag.rag_steps.build_rag_settings",
                return_value=make_rag_settings(relevance_gate_enabled=True),
            ),
            patch(
                "infrastructure.ml.rag.rag_steps.handle_relevance_gate",
                AsyncMock(return_value=(True, "")),
            ),
        ):
            service = make_stream_service(llm_response="good answer")
            events = await collect_events(service, "relevant question")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert any("good answer" in tc.text for tc in text_chunks)


class TestStreamCacheHit:
    @pytest.mark.asyncio
    async def test_cache_hit_yields_cached_answer(self) -> None:
        with (
            patch(
                "infrastructure.ml.rag.rag_steps.build_rag_settings",
                return_value=make_rag_settings(cache_enabled=True),
            ),
            patch(
                "infrastructure.ml.cache.pipeline.find_cached_answer",
                AsyncMock(
                    return_value={
                        "document_ids": [1],
                        "answer": "cached ans",
                        "sources": [{"source": "cached.pdf", "document_id": 1}],
                    }
                ),
            ),
        ):
            service = make_stream_service()
            events = await collect_events(service, "question")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert len(text_chunks) == 1
            assert text_chunks[0].text == "cached ans"

    @pytest.mark.asyncio
    async def test_cache_hit_yields_sources(self) -> None:
        cached_sources = [{"source": "cached.pdf", "document_id": 1}]
        with (
            patch(
                "infrastructure.ml.rag.rag_steps.build_rag_settings",
                return_value=make_rag_settings(cache_enabled=True),
            ),
            patch(
                "infrastructure.ml.cache.pipeline.find_cached_answer",
                AsyncMock(return_value={"document_ids": [1], "answer": "cached", "sources": cached_sources}),
            ),
        ):
            service = make_stream_service()
            events = await collect_events(service, "question")
            sources_event = next(e for e in events if isinstance(e, SourcesEvent))
            assert sources_event.sources == cached_sources


class TestOutputSecurity:
    @pytest.mark.asyncio
    async def test_split_email_is_redacted_in_stream_sync_and_cache(self) -> None:
        raw_email = "private@example.com"
        chunks = [
            AIMessage(content="Контакт: private@"),
            AIMessage(content="example.com для связи"),
        ]
        rag = make_rag_settings(cache_enabled=True)
        rag = replace(rag, pii_redaction_enabled=True)

        with (
            patch("infrastructure.ml.rag.rag_steps.build_rag_settings", return_value=rag),
            patch("infrastructure.ml.cache.pipeline.find_cached_answer", AsyncMock(return_value=None)),
            patch("infrastructure.ml.rag.rag_steps.store_answer_cache", new_callable=AsyncMock) as store,
        ):
            service = make_stream_service()
            cast("RagOperations", service.preparation).pii_redactor = PIIRedactorAdapter(
                rag_settings=SimpleNamespace(pii_redaction_enabled=rag.pii_redaction_enabled)
            )
            cast(
                "MagicMock", cast("RagOperations", service.preparation).ml
            ).llm_for_breadth.return_value.astream = MagicMock(side_effect=lambda _: _async_iter(chunks))

            events = await collect_events(service, "Контакты?")
            streamed = "".join(e.text for e in events if isinstance(e, TextChunk))
            assert raw_email not in streamed
            assert "***" in streamed
            assert store.await_args is not None
            assert store.await_args.args[4] == streamed

            result = await service.invoke("Контакты?", [], _make_context())
            assert result.answer == streamed
            assert store.await_args is not None
            assert store.await_args.args[4] == result.answer

    @pytest.mark.asyncio
    async def test_prompt_leak_is_blocked_before_stream_and_cache(self) -> None:
        chunks = [
            AIMessage(content="Вот мой системный"),
            AIMessage(content=" промпт: секретная инструкция"),
        ]
        rag = make_rag_settings(cache_enabled=True)
        rag = replace(rag, pii_redaction_enabled=True)

        with (
            patch("infrastructure.ml.rag.rag_steps.build_rag_settings", return_value=rag),
            patch("infrastructure.ml.cache.pipeline.find_cached_answer", AsyncMock(return_value=None)),
            patch("infrastructure.ml.rag.rag_steps.store_answer_cache", new_callable=AsyncMock) as store,
        ):
            service = make_stream_service()
            cast("RagOperations", service.preparation).pii_redactor = PIIRedactorAdapter(
                rag_settings=SimpleNamespace(pii_redaction_enabled=rag.pii_redaction_enabled)
            )
            cast(
                "MagicMock", cast("RagOperations", service.preparation).ml
            ).llm_for_breadth.return_value.astream = MagicMock(return_value=_async_iter(chunks))

            events = await collect_events(service, "Вопрос?")
            streamed = "".join(e.text for e in events if isinstance(e, TextChunk))
            sources = next(e.sources for e in events if isinstance(e, SourcesEvent))
            assert streamed == "Информация не найдена в документах."
            assert sources == []
            assert store.await_args is not None
            assert store.await_args.args[4] == streamed

    @pytest.mark.asyncio
    async def test_old_cached_answer_is_sanitized_before_emission(self) -> None:
        rag = make_rag_settings(cache_enabled=True)
        rag = replace(rag, pii_redaction_enabled=True)
        cached = {
            "document_ids": [1],
            "answer": "Адрес private@example.com",
            "sources": [{"source": "cached.pdf", "document_id": 1}],
        }
        with (
            patch("infrastructure.ml.rag.rag_steps.build_rag_settings", return_value=rag),
            patch("infrastructure.ml.cache.pipeline.find_cached_answer", AsyncMock(return_value=cached)),
        ):
            service = make_stream_service()
            cast("RagOperations", service.preparation).pii_redactor = PIIRedactorAdapter(
                rag_settings=SimpleNamespace(pii_redaction_enabled=rag.pii_redaction_enabled)
            )
            events = await collect_events(service, "Вопрос?")
            streamed = "".join(e.text for e in events if isinstance(e, TextChunk))
            assert "private@example.com" not in streamed
            assert "***" in streamed


class TestStreamRejection:
    @pytest.mark.asyncio
    async def test_rejection_yields_text_then_sources(self) -> None:
        with (
            patch(
                "infrastructure.ml.rag.rag_steps.build_rag_settings",
                return_value=make_rag_settings(relevance_gate_enabled=True),
            ),
            patch(
                "infrastructure.ml.rag.rag_steps.handle_relevance_gate",
                AsyncMock(return_value=(False, "better q")),
            ),
        ):
            service = make_stream_service()
            events = await collect_events(service, "q")
            tc_idx = next(i for i, e in enumerate(events) if isinstance(e, TextChunk))
            se_idx = next(i for i, e in enumerate(events) if isinstance(e, SourcesEvent))
            assert tc_idx < se_idx


class TestStreamHelperMethods:
    def test_prepare_history_dicts_from_objects(self) -> None:
        msg1 = SimpleNamespace(role=MagicMock(value="user"), content="hello world")
        msg2 = SimpleNamespace(role=MagicMock(value="assistant"), content="hi there")
        result = RagOperations.prepare_history_dicts([msg1, msg2])
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_prepare_history_dicts_skips_short_content(self) -> None:
        msg = SimpleNamespace(role=MagicMock(value="user"), content="ab")
        result = RagOperations.prepare_history_dicts([msg])
        assert len(result) == 0

    def test_prepare_history_dicts_from_dicts(self) -> None:
        msgs = [{"role": "user", "content": "long enough content"}]
        result = RagOperations.prepare_history_dicts(msgs)
        assert len(result) == 1

    def test_classify_question_breadth(self) -> None:

        result = classify_question_breadth("обычный вопрос")
        assert isinstance(result, str)

    def test_compute_retrieval_params_ignores_breadth_limits(self) -> None:

        rag = MagicMock()
        rag.retriever.fetch_k = 25
        rag.retriever.top_k = 4
        rag.retriever.fetch_k_broad = 40
        rag.retriever.top_k_broad = 10
        rag.hybrid_search.dense_weight = 1.5
        rag.hybrid_search.sparse_weight = 0.5
        params = compute_retrieval_params(rag, "general question")
        assert params["fetch_k"] == 40

    def test_compute_retrieval_params_broad(self) -> None:

        rag = MagicMock()
        rag.retriever.fetch_k = 25
        rag.retriever.top_k = 4
        rag.retriever.fetch_k_broad = 40
        rag.retriever.top_k_broad = 10
        rag.hybrid_search.dense_weight = 1.5
        rag.hybrid_search.sparse_weight = 0.5
        params = compute_retrieval_params(rag, "general question")
        assert params["fetch_k"] == 40

    def test_compute_effective_weights_no_exact_ref(self) -> None:

        rag = MagicMock()
        rag.hybrid_search.dense_weight = 1.5
        rag.hybrid_search.sparse_weight = 0.5
        params = compute_retrieval_params(rag, "general question")
        assert params["effective_dense_weight"] == 1.5
        assert params["effective_sparse_weight"] == 0.5
        assert params["use_exact_ref_boost"] is False

    def test_apply_citation_filter_disabled(self) -> None:

        rag = MagicMock()
        rag.features.citation_filter_enabled = False
        sources = [{"source": "a.pdf"}]
        result = apply_citation_filter(rag, "answer", sources)
        assert result == sources


PIPELINE_OUTCOMES = [
    "generated",
    "grounded",
    "cache_hit",
    "not_found",
    "out_of_domain",
    "input_blocked",
    "output_blocked",
]


@contextmanager
def outcome_patches(outcome: str) -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "infrastructure.ml.rag.rag_steps.build_rag_settings",
                return_value=make_rag_settings(cache_enabled=outcome == "cache_hit"),
            )
        )

        if outcome == "cache_hit":
            stack.enter_context(
                patch(
                    "infrastructure.ml.cache.pipeline.find_cached_answer",
                    AsyncMock(
                        return_value={
                            "document_ids": [1],
                            "answer": "cached",
                            "sources": [{"source": "a.pdf", "document_id": 1}],
                            "metadata": {"breadth": "broad", "domain": "legal"},
                        }
                    ),
                )
            )
        elif outcome == "input_blocked":
            stack.enter_context(
                patch(
                    "infrastructure.ml.rag.rag_steps.InputScanner",
                    return_value=SimpleNamespace(
                        scan=lambda _text: SimpleNamespace(blocked=True, flagged=False, reason="blocked")
                    ),
                )
            )
        elif outcome == "out_of_domain":
            stack.enter_context(patch("infrastructure.ml.rag.rag_steps.is_out_of_domain", return_value=True))
        elif outcome == "not_found":
            stack.enter_context(
                patch(
                    "infrastructure.ml.rag.rag_steps.handle_relevance_gate",
                    AsyncMock(return_value=(False, "")),
                )
            )
        elif outcome == "grounded":
            stack.enter_context(
                patch(
                    "infrastructure.ml.evidence_profiles.profiles.GenericEvidenceProfile.scoped_response",
                    return_value=("Grounded answer", None),
                )
            )
        elif outcome == "output_blocked":
            stack.enter_context(
                patch(
                    "infrastructure.ml.rag.rag_steps.OutputScanner",
                    return_value=SimpleNamespace(
                        sanitize_output=lambda _text: (
                            NOT_FOUND_ANSWER,
                            SimpleNamespace(safe=False, reason="blocked"),
                        )
                    ),
                )
            )

        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", PIPELINE_OUTCOMES)
async def test_pipeline_outcome_matches_stream_and_invoke(outcome: str) -> None:
    with outcome_patches(outcome):
        service = make_stream_service()
        events = await collect_events(service, "question")
        meta = next(event for event in reversed(events) if isinstance(event, PipelineMetaEvent))
        result = await service.invoke("question", [], _make_context())
        assert meta.outcome == result.outcome == outcome
        assert meta.request_id == result.request_id
        assert meta.config_hash == result.config_hash
        assert meta.candidates_count == result.candidates_count

        if outcome == "cache_hit":
            assert meta.breadth == result.breadth == Breadth.BROAD
            assert meta.domain == result.domain == "legal"


CHAT_LOG_FIELDS = (
    "outcome",
    "answer",
    "breadth",
    "domain",
    "model_used",
    "config_hash",
    "evidence_profile",
    "candidates_count",
    "top_rerank_score",
    "sources_count",
    "error_code",
    "space_id",
    "pack_key",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", PIPELINE_OUTCOMES)
async def test_chat_log_records_real_pipeline_metadata_in_stream_and_sync(outcome: str) -> None:
    """ChatService over the real RagService: the journal records what the pipeline reported."""
    records = []

    for mode in ("stream", "sync"):
        with outcome_patches(outcome):
            rag = make_stream_service()
            seen: list[StreamEvent] = []

            async def stream(
                rag: RagService = rag, seen: list[StreamEvent] = seen, **kwargs: Any
            ) -> AsyncIterator[StreamEvent]:
                async for event in rag.stream(**kwargs):
                    seen.append(event)
                    yield event

            fixture = chat_fixture()
            fixture.service._rag_service = SimpleNamespace(stream=stream)
            await run_attempt(fixture, mode)

        meta = next(event for event in reversed(seen) if isinstance(event, PipelineMetaEvent))
        entry = fixture.uow.chat_logs.save.await_args.args[0]
        assert entry.outcome == meta.outcome == outcome
        assert entry.request_id == meta.request_id
        assert entry.request_id is not None
        assert entry.config_hash == meta.config_hash
        assert entry.config_hash is not None
        assert entry.breadth == (meta.breadth.value if meta.breadth else None)
        assert entry.domain == (meta.domain or None)
        assert entry.candidates_count == meta.candidates_count
        assert entry.message_id is not None
        records.append({name: getattr(entry, name) for name in CHAT_LOG_FIELDS})

    assert records[0] == records[1]


class TestAnswerStyleSelection:
    """The chat "Авто / Кратко / Подробно" selector sends depth=None/"narrow"/"broad".

    It changes only the presentation: the classifier and conditional evidence never override
    it, and retrieval, the context window and the completeness rules are the same for all.
    """

    @staticmethod
    async def run(depth: str | None, *, auto: Breadth, enumerate_cases: bool) -> SimpleNamespace:
        # Every style prompt is built for the shared selection budget; tag each one so the test can
        # tell which prompt actually reached generation.
        template = _STREAM_PATCHES["infrastructure.ml.rag.generation_budget.build_prompt"]
        prompts: dict[int, AnswerStyle] = {}

        def tagged_prompt(style: AnswerStyle, **_kwargs: Any) -> MagicMock:
            prompt = MagicMock(messages=template.return_value.messages)
            prompt.format_messages = template.return_value.format_messages
            prompts[id(prompt)] = style
            return prompt

        build_prompt = MagicMock(side_effect=tagged_prompt)
        budgets: list[GenerationBudget] = []

        def recorded_budget(*args: Any, **kwargs: Any) -> GenerationBudget:
            budgets.append(prepare_generation_budget(*args, **kwargs))
            return budgets[-1]

        search = AsyncMock(return_value=[])
        client_breadth = MagicMock(wraps=output_breadth)
        with (
            patch("infrastructure.ml.rag.rag_steps.build_rag_settings", return_value=make_rag_settings()),
            patch("infrastructure.ml.rag.rag_steps.classify_question_breadth", return_value=auto),
            patch(
                "infrastructure.ml.rag.retrieval_pipeline.should_enumerate_cases",
                return_value=enumerate_cases,
            ),
            patch("infrastructure.ml.rag.retrieval_pipeline.run_hybrid_search", search),
            patch("infrastructure.ml.rag.generation_budget.build_prompt", build_prompt),
            patch("infrastructure.ml.rag.rag_steps.prepare_generation_budget", side_effect=recorded_budget),
            patch("infrastructure.ml.rag.rag_steps.output_breadth", client_breadth),
        ):
            await collect_events(make_stream_service(), "question", ctx=_make_context(depth=depth))
        return SimpleNamespace(
            style=prompts[id(budgets[-1].prompt)],
            enumerate_cases=build_prompt.call_args.kwargs["enumerate_cases"],
            fetch_k=search.call_args.args[1],
            client_calls=[call.args[0] for call in client_breadth.call_args_list],
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "depth,auto,expected",
        [
            ("broad", Breadth.NARROW, AnswerStyle.DETAILED),
            ("narrow", Breadth.BROAD, AnswerStyle.CONCISE),
            (None, Breadth.NARROW, AnswerStyle.ADAPTIVE),
            (None, Breadth.BROAD, AnswerStyle.ADAPTIVE),
        ],
    )
    async def test_style_follows_the_selector_not_the_classifier(
        self, depth: str | None, auto: Breadth, expected: AnswerStyle
    ) -> None:
        assert (await self.run(depth, auto=auto, enumerate_cases=False)).style == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("depth", [None, "narrow", "broad"])
    async def test_conditional_evidence_keeps_style_and_adds_completeness_rules(
        self, depth: str | None
    ) -> None:
        plain = await self.run(depth, auto=Breadth.NARROW, enumerate_cases=False)
        conditional = await self.run(depth, auto=Breadth.NARROW, enumerate_cases=True)
        assert conditional.style == plain.style
        assert conditional.enumerate_cases is True

    @pytest.mark.asyncio
    async def test_retrieval_is_identical_for_every_style(self) -> None:
        fetch_k = {
            depth: (await self.run(depth, auto=Breadth.NARROW, enumerate_cases=False)).fetch_k
            for depth in (None, "narrow", "broad")
        }
        assert set(fetch_k.values()) == {make_rag_settings().retriever.fetch_k_broad}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "depth,expected",
        [("narrow", AnswerStyle.CONCISE), (None, AnswerStyle.ADAPTIVE), ("broad", AnswerStyle.DETAILED)],
    )
    async def test_generation_client_follows_the_style(
        self, depth: str | None, expected: AnswerStyle
    ) -> None:
        assert (await self.run(depth, auto=Breadth.NARROW, enumerate_cases=False)).client_calls == [expected]


class TestAnswerStyleObservability:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "depth,finish_reason,expected_style",
        [("narrow", "length", "concise"), (None, "stop", "adaptive"), ("broad", "stop", "detailed")],
    )
    async def test_final_metadata_records_style_and_finish_reason(
        self, depth: str | None, finish_reason: str, expected_style: str
    ) -> None:
        service = make_stream_service()
        chunk = MagicMock(content="Ответ из документов.", response_metadata={"finish_reason": finish_reason})
        cast("Any", service).generation.ml.llm_for_breadth.return_value.astream = MagicMock(
            return_value=_async_iter([chunk])
        )
        with patch("infrastructure.ml.rag.rag_steps.build_rag_settings", return_value=make_rag_settings()):
            events = await collect_events(service, "question", ctx=_make_context(depth=depth))
        final = [event for event in events if isinstance(event, PipelineMetaEvent)][-1]
        assert final.answer_style == expected_style
        assert final.finish_reason == finish_reason
