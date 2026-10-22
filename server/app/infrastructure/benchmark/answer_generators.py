"""Adapt full RAG execution to the same benchmark answer contract."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from application.chat import ChatRAGPort
from config import _settings_overrides
from domain.rag.context import RagContext

from infrastructure.ml.rag.benchmark_evidence import (
    BenchmarkEvidence,
    active_evidence,
)

from infrastructure.benchmark.case_metrics import (
    compute_retriever_metrics_from_sources,
)
from infrastructure.ml.usage_capture import active_llm_usage


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
    rag_latency_sec: float | None = None
    llm_usage: list[dict] | None = None


class BenchmarkAnswerGenerator(Protocol):
    async def generate(self, question: dict, ctx: RagContext) -> BenchmarkAnswer: ...


class RagBenchmarkGenerator:
    def __init__(self, rag: ChatRAGPort, top_k: int, fetch_k: int | None) -> None:
        self._rag = rag
        self._top_k = top_k
        self._fetch_k = fetch_k

    async def generate(self, question: dict, ctx: RagContext) -> BenchmarkAnswer:
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
        records = active_llm_usage.get()
        if records is None:
            records = []
        usage_token = active_llm_usage.set(records)
        started = time.monotonic()
        try:
            result = await self._rag.invoke(question=question["question"], history=[], ctx=ctx)
            elapsed = time.monotonic() - started
        finally:
            active_llm_usage.reset(usage_token)
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
            rag_latency_sec=elapsed,
            llm_usage=records,
        )
