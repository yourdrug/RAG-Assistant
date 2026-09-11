"""Characterization tests for RagService.stream() — the 270-line god-method.

Locks down the full RAG pipeline orchestration BEFORE refactoring.
Every test mocks external I/O (Qdrant, Ollama, Redis) and verifies
the event sequence produced by stream().
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from domain.value_objects.chat_context import ChatContext  # noqa: E402
from domain.value_objects.llm_provider import Breadth  # noqa: E402
from domain.value_objects.stream_events import SourcesEvent, StatusEvent, TextChunk  # noqa: E402
from infrastructure.ml.rag_service import RagService  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(**overrides) -> ChatContext:
    defaults = {
        "user_id": 1,
        "user_kind": "internal",
        "user_role": "admin",
        "user_group_ids": [10],
        "depth": None,
        "summary": None,
        "as_of_date": None,
    }
    defaults.update(overrides)
    return ChatContext(**defaults)


async def _async_iter(items):
    for item in items:
        yield item


def _make_rag_settings(**feature_overrides):
    features = MagicMock(
        condense_enabled=False,
        decomposition_enabled=False,
        cache_enabled=False,
        relevance_gate_enabled=False,
        citation_filter_enabled=False,
        rolling_summary_enabled=False,
    )
    for k, v in feature_overrides.items():
        setattr(features, k, v)
    return MagicMock(
        features=features,
        retriever=MagicMock(fetch_k=25, top_k=4, fetch_k_broad=40, top_k_broad=10),
        hybrid_search=MagicMock(enabled=False, dense_weight=1.0, sparse_weight=1.0, rrf_k=30),
        rerank=MagicMock(min_score=0.0, score_gap_ratio=0.0),
        source_min_score=0.0,
    )


def _make_service(llm_response: str = "Ответ из документов.", **overrides):
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
    mock_ml.bm25_index.return_value = None

    chunk_search = overrides.pop("chunk_search", None)
    domain_registry = overrides.pop("domain_registry", None)
    return RagService(ml_clients=mock_ml, chunk_search=chunk_search, domain_registry=domain_registry)


async def collect_events(service, question, history=None, ctx=None):
    if history is None:
        history = []
    if ctx is None:
        ctx = _make_context()
    events = []
    async for event in service.stream(question, history, ctx):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Patch all I/O for stream() tests
# ---------------------------------------------------------------------------

_STREAM_PATCHES = {
    "infrastructure.ml.rag_service.build_qdrant_filter": MagicMock(return_value=MagicMock(should=[])),
    "infrastructure.ml.rag_service.with_temporal_filter": MagicMock(return_value=MagicMock(should=[])),
    "infrastructure.ml.rag.rag_steps.compute_visibility_scope_hash": MagicMock(return_value="vis_hash"),
    "infrastructure.ml.rag.rag_steps.compute_question_hash": MagicMock(return_value="q_hash"),
    "infrastructure.ml.rag.rag_steps.classify_query_domain": MagicMock(return_value="general"),
    "infrastructure.ml.rag.rag_steps.is_out_of_domain": MagicMock(return_value=False),
    "infrastructure.ml.rag.rag_steps.condense_question": AsyncMock(return_value="condensed question"),
    "infrastructure.ml.rag.rag_steps.rerank_documents": AsyncMock(
        return_value=[
            (SimpleNamespace(page_content="chunk1", metadata={"source": "a.pdf"}), 0.9),
            (SimpleNamespace(page_content="chunk2", metadata={"source": "a.pdf"}), 0.8),
        ]
    ),
    "infrastructure.ml.rag.rag_steps.deduplicate_docs": lambda docs: docs,
    "infrastructure.ml.rag.rag_steps.group_by_section": lambda docs: docs,
    "infrastructure.ml.rag.rag_steps.build_prompt": MagicMock(
        return_value=MagicMock(
            messages=[MagicMock(content="system prompt")],
            format_messages=lambda **kw: [MagicMock(content="user msg")],
        )
    ),
    "infrastructure.ml.rag.rag_steps.format_docs": MagicMock(return_value="formatted context"),
    "infrastructure.ml.rag.rag_steps.extract_sources": MagicMock(
        return_value=[{"source": "a.pdf", "score": 0.9}]
    ),
    "infrastructure.ml.rag.rag_steps.apply_citation_filter": lambda rag, answer, sources: sources,
    "infrastructure.ml.rag.rag_steps.record_rag_answer": MagicMock(),
    "infrastructure.ml.rag.rag_steps.record_llm_usage": MagicMock(),
    "infrastructure.ml.rag.rag_steps.extract_usage_from_langchain": MagicMock(return_value=(100, 50)),
    "infrastructure.ml.rag.rag_steps.request_id_ctx": MagicMock(
        get=MagicMock(return_value="test-req-id"), set=MagicMock()
    ),
    "infrastructure.ml.rag.rag_steps.classify_question_breadth": MagicMock(return_value="narrow"),
    "infrastructure.ml.rag.rag_steps.has_exact_reference": MagicMock(return_value=False),
    "infrastructure.ml.rag.rag_steps.should_enumerate_cases": MagicMock(return_value=False),
}


@pytest.fixture(autouse=True)
def _patch_stream_deps():
    """Patch all I/O-dependent imports for RagService.stream()."""
    patchers = []
    for target, mock_val in _STREAM_PATCHES.items():
        p = patch(target, mock_val)
        p.start()
        patchers.append(p)
    yield
    for p in patchers:
        p.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamEventSequence:
    @pytest.mark.asyncio
    async def test_yields_status_searching(self):
        with patch(
            "infrastructure.ml.rag_service._build_rag_settings",
            return_value=_make_rag_settings(),
        ):
            service = _make_service(llm_response="test answer")
            events = await collect_events(service, "What is X?")
            stages = [e.stage for e in events if isinstance(e, StatusEvent)]
            assert "searching" in stages

    @pytest.mark.asyncio
    async def test_yields_status_reranking(self):
        with patch(
            "infrastructure.ml.rag_service._build_rag_settings",
            return_value=_make_rag_settings(),
        ):
            service = _make_service()
            events = await collect_events(service, "What is X?")
            stages = [e.stage for e in events if isinstance(e, StatusEvent)]
            assert "reranking" in stages

    @pytest.mark.asyncio
    async def test_yields_status_generating(self):
        with patch(
            "infrastructure.ml.rag_service._build_rag_settings",
            return_value=_make_rag_settings(),
        ):
            service = _make_service()
            events = await collect_events(service, "What is X?")
            stages = [e.stage for e in events if isinstance(e, StatusEvent)]
            assert "generating" in stages

    @pytest.mark.asyncio
    async def test_yields_text_chunk(self):
        with patch(
            "infrastructure.ml.rag_service._build_rag_settings",
            return_value=_make_rag_settings(),
        ):
            service = _make_service(llm_response="Hello world")
            events = await collect_events(service, "question")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert len(text_chunks) >= 1
            assert text_chunks[0].text == "Hello world"

    @pytest.mark.asyncio
    async def test_yields_sources_event_last(self):
        with patch(
            "infrastructure.ml.rag_service._build_rag_settings",
            return_value=_make_rag_settings(),
        ):
            service = _make_service()
            events = await collect_events(service, "question")
            assert isinstance(events[-1], SourcesEvent)

    @pytest.mark.asyncio
    async def test_sources_event_has_sources(self):
        with patch(
            "infrastructure.ml.rag_service._build_rag_settings",
            return_value=_make_rag_settings(),
        ):
            service = _make_service()
            events = await collect_events(service, "question")
            sources_event = [e for e in events if isinstance(e, SourcesEvent)][0]
            assert isinstance(sources_event.sources, list)


class TestStreamOutOfDomain:
    @pytest.mark.asyncio
    async def test_out_of_domain_yields_not_found(self):
        patch("infrastructure.ml.rag.rag_steps.is_out_of_domain", MagicMock(return_value=True)).start()
        with patch(
            "infrastructure.ml.rag_service._build_rag_settings",
            return_value=_make_rag_settings(),
        ):
            service = _make_service()
            events = await collect_events(service, "tell me a joke")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert len(text_chunks) == 1
            assert "не найдена" in text_chunks[0].text.lower()

    @pytest.mark.asyncio
    async def test_out_of_domain_yields_empty_sources(self):
        patch("infrastructure.ml.rag.rag_steps.is_out_of_domain", MagicMock(return_value=True)).start()
        with patch(
            "infrastructure.ml.rag_service._build_rag_settings",
            return_value=_make_rag_settings(),
        ):
            service = _make_service()
            events = await collect_events(service, "joke?")
            sources_event = [e for e in events if isinstance(e, SourcesEvent)][0]
            assert sources_event.sources == []

    @pytest.mark.asyncio
    async def test_out_of_domain_does_not_call_llm(self):
        patch("infrastructure.ml.rag.rag_steps.is_out_of_domain", MagicMock(return_value=True)).start()
        with patch(
            "infrastructure.ml.rag_service._build_rag_settings",
            return_value=_make_rag_settings(),
        ):
            service = _make_service()
            await collect_events(service, "joke?")
            service._ml.llm_for_breadth.return_value.astream.assert_not_called()


class TestStreamRelevanceGate:
    @pytest.mark.asyncio
    async def test_rejection_yields_not_relevant_message(self):
        with (
            patch(
                "infrastructure.ml.rag_service._build_rag_settings",
                return_value=_make_rag_settings(relevance_gate_enabled=True),
            ),
            patch(
                "infrastructure.ml.rag.rag_steps.handle_relevance_gate",
                AsyncMock(return_value=False),
            ),
            patch(
                "infrastructure.ml.rag.rag_steps._assess_sufficiency",
                AsyncMock(return_value=MagicMock(is_sufficient=False, suggested_refinement="better query")),
            ),
        ):
            service = _make_service()
            events = await collect_events(service, "obscure question")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert len(text_chunks) == 1
            assert "не нашёл релевантной информации" in text_chunks[0].text

    @pytest.mark.asyncio
    async def test_pass_yields_answer(self):
        with (
            patch(
                "infrastructure.ml.rag_service._build_rag_settings",
                return_value=_make_rag_settings(relevance_gate_enabled=True),
            ),
            patch(
                "infrastructure.ml.rag.rag_steps.handle_relevance_gate",
                AsyncMock(return_value=True),
            ),
        ):
            service = _make_service(llm_response="good answer")
            events = await collect_events(service, "relevant question")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert any("good answer" in tc.text for tc in text_chunks)


class TestStreamCacheHit:
    @pytest.mark.asyncio
    async def test_cache_hit_yields_cached_answer(self):
        with (
            patch(
                "infrastructure.ml.rag_service._build_rag_settings",
                return_value=_make_rag_settings(cache_enabled=True),
            ),
            patch(
                "infrastructure.ml.rag.rag_cache.find_cached_answer",
                AsyncMock(return_value={"answer": "cached ans", "sources": [{"source": "cached.pdf"}]}),
            ),
        ):
            service = _make_service()
            events = await collect_events(service, "question")
            text_chunks = [e for e in events if isinstance(e, TextChunk)]
            assert len(text_chunks) == 1
            assert text_chunks[0].text == "cached ans"

    @pytest.mark.asyncio
    async def test_cache_hit_yields_sources(self):
        cached_sources = [{"source": "cached.pdf"}]
        with (
            patch(
                "infrastructure.ml.rag_service._build_rag_settings",
                return_value=_make_rag_settings(cache_enabled=True),
            ),
            patch(
                "infrastructure.ml.rag.rag_cache.find_cached_answer",
                AsyncMock(return_value={"answer": "cached", "sources": cached_sources}),
            ),
        ):
            service = _make_service()
            events = await collect_events(service, "question")
            sources_event = [e for e in events if isinstance(e, SourcesEvent)][0]
            assert sources_event.sources == cached_sources


class TestStreamRejection:
    @pytest.mark.asyncio
    async def test_rejection_yields_text_then_sources(self):
        with (
            patch(
                "infrastructure.ml.rag_service._build_rag_settings",
                return_value=_make_rag_settings(relevance_gate_enabled=True),
            ),
            patch(
                "infrastructure.ml.rag.rag_steps.handle_relevance_gate",
                AsyncMock(return_value=False),
            ),
            patch(
                "infrastructure.ml.rag.rag_steps._assess_sufficiency",
                AsyncMock(return_value=MagicMock(is_sufficient=False, suggested_refinement="better q")),
            ),
        ):
            service = _make_service()
            events = await collect_events(service, "q")
            tc_idx = next(i for i, e in enumerate(events) if isinstance(e, TextChunk))
            se_idx = next(i for i, e in enumerate(events) if isinstance(e, SourcesEvent))
            assert tc_idx < se_idx


class TestStreamHelperMethods:
    def test_prepare_history_dicts_from_objects(self):
        msg1 = SimpleNamespace(role=MagicMock(value="user"), content="hello world")
        msg2 = SimpleNamespace(role=MagicMock(value="assistant"), content="hi there")
        result = RagService._prepare_history_dicts([msg1, msg2])
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_prepare_history_dicts_skips_short_content(self):
        msg = SimpleNamespace(role=MagicMock(value="user"), content="ab")
        result = RagService._prepare_history_dicts([msg])
        assert len(result) == 0

    def test_prepare_history_dicts_from_dicts(self):
        msgs = [{"role": "user", "content": "long enough content"}]
        result = RagService._prepare_history_dicts(msgs)
        assert len(result) == 1

    def test_resolve_breadth_from_context(self):
        service = _make_service()
        ctx = _make_context(depth="broad")
        with patch(
            "domain.value_objects.llm_provider.BREADTH_ALIASES",
            {"broad": Breadth.BROAD},
        ):
            breadth = service._resolve_breadth(ctx, "question")
            assert breadth == Breadth.BROAD

    def test_resolve_fetch_top_k_narrow(self):
        service = _make_service()
        rag = MagicMock()
        rag.retriever.fetch_k = 25
        rag.retriever.top_k = 4
        rag.retriever.fetch_k_broad = 40
        rag.retriever.top_k_broad = 10
        fetch_k, top_k = service._resolve_fetch_top_k(rag, Breadth.NARROW)
        assert fetch_k == 25
        assert top_k == 4

    def test_resolve_fetch_top_k_broad(self):
        service = _make_service()
        rag = MagicMock()
        rag.retriever.fetch_k = 25
        rag.retriever.top_k = 4
        rag.retriever.fetch_k_broad = 40
        rag.retriever.top_k_broad = 10
        fetch_k, top_k = service._resolve_fetch_top_k(rag, Breadth.BROAD)
        assert fetch_k == 40
        assert top_k == 10

    def test_compute_effective_weights_no_exact_ref(self):
        service = _make_service()
        rag = MagicMock()
        rag.hybrid_search.dense_weight = 1.5
        rag.hybrid_search.sparse_weight = 0.5
        with patch(
            "domain.services.rag_policy.has_exact_reference",
            MagicMock(return_value=False),
        ):
            dense, sparse, exact = service._compute_effective_weights(rag, "general question")
            assert dense == 1.5
            assert sparse == 0.5
            assert exact is False

    def test_apply_citation_filter_disabled(self):
        from infrastructure.ml.rag.rag_postprocess import apply_citation_filter

        rag = MagicMock()
        rag.features.citation_filter_enabled = False
        sources = [{"source": "a.pdf"}]
        result = apply_citation_filter(rag, "answer", sources)
        assert result == sources

    def test_resolve_temporal_conflicts_no_conflicts(self):
        from infrastructure.ml.rag.rag_postprocess import resolve_temporal_conflicts

        doc1 = SimpleNamespace(metadata={"source": "a.pdf"})
        doc2 = SimpleNamespace(metadata={"source": "b.pdf"})
        docs = [(doc1, 0.9), (doc2, 0.8)]
        result = resolve_temporal_conflicts(docs)
        assert len(result) == 2

    def test_resolve_temporal_conflicts_keeps_latest(self):
        from infrastructure.ml.rag.rag_postprocess import resolve_temporal_conflicts

        doc_old = SimpleNamespace(metadata={"act_id": 1, "act_version_id": 1, "effective_from": "2020-01-01"})
        doc_new = SimpleNamespace(metadata={"act_id": 1, "act_version_id": 2, "effective_from": "2024-01-01"})
        docs = [(doc_old, 0.9), (doc_new, 0.8)]
        result = resolve_temporal_conflicts(docs)
        assert len(result) == 1
        assert result[0][0].metadata["act_version_id"] == 2
