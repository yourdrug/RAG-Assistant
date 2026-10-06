"""Adapt full and standalone RAG execution to the same benchmark answer contract."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from application.ports.chat_rag_port import ChatRAGPort
from config import _settings_overrides, get_setting, settings
from domain.value_objects.chat_context import ChatContext

from infrastructure.ml.rag.benchmark_evidence import (
    BenchmarkEvidence,
    active_evidence,
    snapshot_documents,
)

from infrastructure.benchmark.case_metrics import (
    compute_retriever_metrics_from_sources,
    _extract_usage_from_response,
)


@dataclass(frozen=True)
class BenchmarkAnswer:
    answer: str
    context: str
    retriever_metrics: dict
    input_tokens: int | None
    output_tokens: int | None
    ttft_sec: float | None = None
    breadth: str | None = None
    domain: str | None = None
    evidence: BenchmarkEvidence | None = None


class BenchmarkAnswerGenerator(Protocol):
    async def generate(self, question: dict, ctx: ChatContext) -> BenchmarkAnswer: ...


class RagBenchmarkGenerator:
    def __init__(self, rag: ChatRAGPort, top_k: int, fetch_k: int | None) -> None:
        self._rag = rag
        self._top_k = top_k
        self._fetch_k = fetch_k

    async def generate(self, question: dict, ctx: ChatContext) -> BenchmarkAnswer:
        overrides = {
            **(_settings_overrides.get() or {}),
            "cache_enabled": False,
            "retriever_top_k": self._top_k,
        }
        if self._fetch_k is not None:
            overrides["retriever_fetch_k"] = self._fetch_k
        evidence = BenchmarkEvidence()
        evidence_token = active_evidence.set(evidence)
        token = _settings_overrides.set(overrides)
        try:
            result = await self._rag.invoke(question=question["question"], history=[], ctx=ctx)
        finally:
            _settings_overrides.reset(token)
            active_evidence.reset(evidence_token)
        context = "\n\n---\n\n".join(s.get("content", "") for s in result.sources if s.get("content"))
        if evidence.context is not None:
            context = evidence.context
        return BenchmarkAnswer(
            answer=result.answer,
            context=context,
            retriever_metrics=compute_retriever_metrics_from_sources(
                result.sources, question.get("source_hint")
            ),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            ttft_sec=result.ttft_sec,
            breadth=result.breadth,
            domain=result.domain,
            evidence=evidence,
        )


class StandaloneBenchmarkGenerator:
    def __init__(self, top_k: int, fetch_k: int | None, ctx: ChatContext) -> None:
        from domain.services.access_control import get_visibility_conditions
        from domain.value_objects.roles import UserKind, UserRole

        from infrastructure.repositories.vector.acl import build_qdrant_filter

        self._top_k = top_k
        self._fetch_k = fetch_k
        self._access_filter = build_qdrant_filter(
            user={"id": ctx.user_id, "kind": UserKind.INTERNAL, "role": UserRole.ADMIN}, group_ids=[]
        )
        self._visibility_conditions = get_visibility_conditions(
            user_kind=UserKind.INTERNAL,
            user_id=ctx.user_id,
            group_ids=[],
            for_list=False,
            user_role=UserRole.ADMIN,
        )

    async def generate(self, question: dict, ctx: ChatContext) -> BenchmarkAnswer:
        from infrastructure.benchmark.judge import get_rag_answer_with_usage
        from infrastructure.benchmark.metrics import compute_retriever_metrics
        from infrastructure.benchmark.retrieval import build_llm, retrieve_with_scores_hybrid

        llm = build_llm(settings.llm_model, settings.ollama_base_url, provider=settings.llm_provider)
        docs = await asyncio.to_thread(
            retrieve_with_scores_hybrid,
            question["question"],
            self._top_k,
            self._fetch_k or int(get_setting("rag.retriever_fetch_k")),
            access_filter=self._access_filter,
            visibility_conditions=self._visibility_conditions,
            user_id=ctx.user_id,
            user_group_ids=ctx.user_group_ids,
        )
        from infrastructure.ml.rag.rag_formatting import format_docs_with_selection

        context, selected = format_docs_with_selection(docs, max_context_tokens=4000)
        answer, response = await asyncio.to_thread(get_rag_answer_with_usage, llm, docs, question["question"])
        input_tokens, output_tokens = _extract_usage_from_response(response)
        return BenchmarkAnswer(
            answer=answer,
            context=context,
            evidence=BenchmarkEvidence(snapshot_documents(docs), snapshot_documents(selected), context),
            retriever_metrics=compute_retriever_metrics(docs, question.get("source_hint")),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
