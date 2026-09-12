"""Infrastructure implementation of the ChatRAGPort used by ChatService.

Handles the full RAG pipeline: question classification, context retrieval
(Qdrant + BM25 hybrid), reranking, prompt assembly, and LLM streaming.
Exposes ``stream_answer`` as an async iterator of tagged union events
(``TextChunk | SourcesEvent``).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from domain.value_objects.rag_result import RagResult
from config import settings
from domain.utils import compute_reranker_score
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.stream_events import (
    PipelineMetaEvent,
    SourcesEvent,
    StatusEvent,
    StreamEvent,
    TextChunk,
    UsageReport,
)

if TYPE_CHECKING:
    from infrastructure.ml.clients.client_registry import MLClientRegistry

from infrastructure.ml.rag.rag_cache import store_answer_cache
from infrastructure.ml.rag.rag_config import build_rag_settings
from infrastructure.ml.rag.rag_formatting import history_to_messages
from infrastructure.ml.rag.rag_steps import (
    step_build_context,
    step_check_cache,
    step_condense,
    step_generate,
    step_postprocess,
    step_reject_ood,
    step_retrieve,
    step_self_rag,
)
from infrastructure.ml.rag_pipeline import RagPipelineState
from infrastructure.repositories.vector.acl import build_qdrant_filter, with_temporal_filter
from shared import request_id_ctx

# Backward-compat re-exports (used by test_rag_pipeline.py, test_rag_service_characterization.py)
from infrastructure.ml.rag.rag_config import build_rag_settings as _build_rag_settings  # noqa: F401
from infrastructure.ml.rag.rag_postprocess import is_not_found_answer as _is_not_found_answer  # noqa: F401

log = logging.getLogger("default")


class RagService:
    def __init__(self, ml_clients: MLClientRegistry, chunk_search=None, domain_registry=None) -> None:
        self._ml = ml_clients
        self._chunk_search = chunk_search
        self._domain_registry = domain_registry

    @staticmethod
    def _prepare_history_dicts(history: list) -> list[dict]:
        history_dicts = []
        for msg in history:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                content = msg.content
                if len(content.strip()) < 3:
                    continue
                history_dicts.append(
                    {
                        "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                        "content": content,
                    }
                )
            elif isinstance(msg, dict):
                if len(msg.get("content", "").strip()) >= 3:
                    history_dicts.append(msg)
        return history_dicts

    # ── Backward-compat thin wrappers (used by characterization tests) ───

    def _resolve_breadth(self, ctx: ChatContext, query_for_search: str):
        from domain.value_objects.llm_provider import BREADTH_ALIASES
        from domain.services.rag_policy import classify_question_breadth
        from infrastructure.metrics.metrics import RAG_BREADTH_TOTAL

        raw = ctx.depth if ctx.depth in BREADTH_ALIASES else classify_question_breadth(query_for_search)
        breadth = BREADTH_ALIASES.get(raw) or Breadth(raw)
        RAG_BREADTH_TOTAL.labels(breadth=breadth).inc()
        return breadth

    def _compute_effective_weights(self, rag, query_for_search: str):
        from domain.services.rag_policy import has_exact_reference

        use_exact_ref_boost = has_exact_reference(query_for_search)
        effective_dense_weight = rag.hybrid_search.dense_weight
        effective_sparse_weight = rag.hybrid_search.sparse_weight
        if use_exact_ref_boost:
            effective_sparse_weight = rag.hybrid_search.sparse_weight * settings.exact_ref_sparse_boost
        return effective_dense_weight, effective_sparse_weight, use_exact_ref_boost

    def _resolve_fetch_top_k(self, rag, breadth: Breadth):
        fetch_k = rag.retriever.fetch_k_broad if breadth == Breadth.BROAD else rag.retriever.fetch_k
        top_k = rag.retriever.top_k_broad if breadth == Breadth.BROAD else rag.retriever.top_k
        return fetch_k, top_k

    def _init_state(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> RagPipelineState:
        """Build the initial pipeline state from request parameters."""
        import time

        rag = build_rag_settings()
        t_pipeline_start = time.monotonic()

        req_id = request_id_ctx.get("")
        if not req_id:
            req_id = uuid.uuid4().hex[:12]
            request_id_ctx.set(req_id)

        user = {"id": ctx.user_id, "kind": ctx.user_kind, "role": ctx.user_role}
        access_filter = build_qdrant_filter(user, ctx.user_group_ids)
        retrieval_filter = with_temporal_filter(access_filter, ctx.as_of_date)

        history_dicts = self._prepare_history_dicts(history)
        history_messages = history_to_messages(history_dicts)

        return RagPipelineState(
            rag=rag,
            t_pipeline_start=t_pipeline_start,
            question=question,
            ctx=ctx,
            user=user,
            access_filter=access_filter,
            retrieval_filter=retrieval_filter,
            req_id=req_id,
            history_messages=history_messages,
        )

    async def stream(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> AsyncIterator[StreamEvent]:
        state = self._init_state(question, history, ctx)

        # ── Step 1: Query condensation ──────────────────────────────
        state = await step_condense(state, self._ml)

        # ── Step 2: Semantic answer cache ───────────────────────────
        state, events = await step_check_cache(state)
        if state.terminal:
            async for e in events:
                yield e
            return

        # ── Step 3: Out-of-domain rejection ─────────────────────────
        state, events = await step_reject_ood(state)
        if state.terminal:
            async for e in events:
                yield e
            return

        # ── Step 4: Retrieval pipeline ──────────────────────────────
        yield StatusEvent(stage="searching")
        state = await step_retrieve(state, self._ml, self._chunk_search)
        yield StatusEvent(stage="reranking")

        # ── Step 5: Self-RAG relevance gate + retry loop ────────────
        state, events = await step_self_rag(state, self._ml, self._chunk_search)
        if state.terminal:
            async for e in events:
                yield e
            return

        # ── Step 6: Prompt building + LLM generation ────────────────
        yield StatusEvent(stage="generating")
        messages, grouped_docs = step_build_context(state, self._domain_registry)
        async for event in step_generate(state, self._ml, messages, grouped_docs):
            yield event

        # ── Step 7: Post-processing + metrics + cache + final yield ─
        state = step_postprocess(state)

        if state.rag.features.cache_enabled:
            await store_answer_cache(
                state.docs,
                state.query_for_search,
                state.q_hash,
                state.vis_hash,
                state.full_answer,
                state.sources,
            )

        yield SourcesEvent(
            sources=state.sources,
            confidence=state.confidence,
            usage=state.usage_report if isinstance(state.usage_report, UsageReport) else None,
        )
        yield PipelineMetaEvent(
            breadth=state.breadth,
            domain=state.query_domain,
            ttft_sec=state.ttft_sec,
        )

    async def invoke(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> RagResult:
        answer_parts: list[str] = []
        sources: list[dict] = []
        breadth = Breadth.NARROW
        domain = DocDomain.GENERAL.value
        retrieval_count = 0
        reranker_score: float | None = None
        confidence: float | None = None
        usage_report = None
        ttft_sec: float | None = None

        async for event in self.stream(question, history, ctx):
            if isinstance(event, SourcesEvent):
                sources = event.sources
                confidence = event.confidence
                usage_report = event.usage
            elif isinstance(event, PipelineMetaEvent):
                if event.breadth is not None:
                    breadth = event.breadth
                if event.domain:
                    domain = event.domain
                if event.ttft_sec is not None:
                    ttft_sec = event.ttft_sec
            elif isinstance(event, TextChunk):
                answer_parts.append(event.text)

        retrieval_count = len(sources)
        if sources:
            reranker_score = compute_reranker_score(sources)

        return RagResult(
            answer="".join(answer_parts),
            sources=sources,
            breadth=breadth,
            domain=domain,
            retrieval_count=retrieval_count,
            reranker_score=reranker_score,
            confidence=confidence,
            model_used=settings.llm_model,
            input_tokens=usage_report.input_tokens if usage_report else None,
            output_tokens=usage_report.output_tokens if usage_report else None,
            ttft_sec=ttft_sec,
        )
