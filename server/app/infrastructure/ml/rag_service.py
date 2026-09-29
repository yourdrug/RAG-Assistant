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

from application.ports.ml_clients import MLClientPort

from config import settings
from domain.value_objects.rag_result import RagResult
from domain.utils import compute_reranker_score
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.user_context import UserContext
from domain.value_objects.stream_events import (
    PipelineMetaEvent,
    SourcesEvent,
    StatusEvent,
    StreamEvent,
    TextChunk,
    UsageReport,
)

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
    step_scan_input,
    step_self_rag,
)
from infrastructure.ml.rag_pipeline import RagPipelineState
from infrastructure.repositories.vector.acl import build_qdrant_filter, with_temporal_filter
from shared import request_id_ctx

if TYPE_CHECKING:
    from application.ports.pii_redactor import PIIRedactorPort
    from domain.domain_profile.registry import DomainProfileRegistry
    from infrastructure.adapters.chunk_search_adapter import ChunkSearchAdapter

log = logging.getLogger("default")


class RagService:
    def __init__(
        self,
        ml_clients: MLClientPort,
        chunk_search: ChunkSearchAdapter | None = None,
        domain_registry: DomainProfileRegistry | None = None,
        pii_redactor: PIIRedactorPort | None = None,
    ) -> None:
        self._ml = ml_clients
        self._chunk_search = chunk_search
        self._domain_registry = domain_registry
        self._pii_redactor = pii_redactor

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

        user = UserContext(user_id=ctx.user_id, user_kind=ctx.user_kind, user_role=ctx.user_role)
        scope = ctx.curator_scope

        from domain.services import get_visibility_conditions
        from domain.value_objects.roles import UserKind, UserRole

        visibility_conditions = get_visibility_conditions(
            UserKind(ctx.user_kind),
            ctx.user_id,
            ctx.user_group_ids,
            for_list=False,
            user_role=UserRole(ctx.user_role) if ctx.user_role else None,
            managed_client_ids=list(scope.managed_client_ids) if scope else None,
            managed_internal_ids=list(scope.managed_internal_ids) if scope else None,
            managed_group_ids=list(scope.managed_group_ids) if scope else None,
        )

        access_filter = build_qdrant_filter(
            user,
            ctx.user_group_ids,
            managed_client_ids=list(scope.managed_client_ids) if scope else None,
            managed_internal_ids=list(scope.managed_internal_ids) if scope else None,
            managed_group_ids=list(scope.managed_group_ids) if scope else None,
        )
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
            visibility_conditions=visibility_conditions,
            req_id=req_id,
            history_messages=history_messages,
            pii_redactor=self._pii_redactor,
        )

    @staticmethod
    async def _yield_events(events: AsyncIterator[StreamEvent] | None) -> AsyncIterator[StreamEvent]:
        """Yield all events from an async iterator if it's not None."""
        if events is not None:
            async for e in events:
                yield e

    async def _drain_if_terminal(
        self,
        state: RagPipelineState,
        events: AsyncIterator[StreamEvent] | None,
    ) -> list[StreamEvent]:
        """Return terminal events as a list, or empty list if not terminal."""
        if not state.terminal:
            return []
        return [e async for e in self._yield_events(events)]

    async def stream(  # noqa: C901 — pipeline orchestrator, inherent complexity
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> AsyncIterator[StreamEvent]:
        state = self._init_state(question, history, ctx)

        # ── Step 1: Query condensation ──────────────────────────────
        state = await step_condense(state, self._ml)

        # ── Step 1.5: Input injection scanning ──────────────────────
        state, events = await step_scan_input(state)
        for e in await self._drain_if_terminal(state, events):
            yield e
        if state.terminal:
            return

        # ── Step 2: Semantic answer cache ───────────────────────────
        state, events = await step_check_cache(state)
        for e in await self._drain_if_terminal(state, events):
            yield e
        if state.terminal:
            return

        # ── Step 3: Out-of-domain rejection ─────────────────────────
        state, events = await step_reject_ood(state)
        for e in await self._drain_if_terminal(state, events):
            yield e
        if state.terminal:
            return

        # ── Step 4: Retrieval pipeline ──────────────────────────────
        yield StatusEvent(stage="searching")
        state = await step_retrieve(state, self._ml, self._chunk_search)
        yield StatusEvent(stage="reranking")

        # ── Step 5: Self-RAG relevance gate + retry loop ────────────
        state, events = await step_self_rag(state, self._ml, self._chunk_search)
        for e in await self._drain_if_terminal(state, events):
            yield e
        if state.terminal:
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
