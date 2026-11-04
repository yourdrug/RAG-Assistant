"""Adapt full RAG execution to the same benchmark answer contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
import time
from typing import TYPE_CHECKING, Any, Protocol, cast

from config import _settings_overrides
from domain.rag.exceptions import ContextBudgetExceededError
from infrastructure.benchmark.evaluation.case_metrics import (
    compute_retriever_metrics_from_sources,
)
from infrastructure.ml.rag.benchmark_evidence import (
    BenchmarkEvidence,
    active_evidence,
)
from infrastructure.ml.usage_capture import active_llm_usage


if TYPE_CHECKING:
    from application.chat import ChatRAGPort
    from domain.rag.context import RagContext


@dataclass(frozen=True)
class BenchmarkAnswer:
    answer: str
    context: str
    retriever_metrics: dict[str, Any]
    input_tokens: int | None
    output_tokens: int | None
    ttft_sec: float | None = None
    breadth: str | None = None
    domain: str | None = None
    evidence: BenchmarkEvidence | None = None
    rag_latency_sec: float | None = None
    llm_usage: list[dict[str, Any]] | None = None


class EvidenceFailure(Protocol):
    benchmark_evidence: BenchmarkEvidence


class BenchmarkAnswerGenerator(Protocol):
    async def generate(self, question: dict[str, Any], ctx: RagContext) -> BenchmarkAnswer: ...


class RagBenchmarkGenerator:
    def __init__(self, rag: ChatRAGPort, top_k: int, fetch_k: int | None) -> None:
        self._rag = rag
        self._top_k = top_k
        self._fetch_k = fetch_k

    async def generate(self, question: dict[str, Any], ctx: RagContext) -> BenchmarkAnswer:
        overrides = {
            **(_settings_overrides.get() or {}),
            "cache_enabled": False,
            "retriever_top_k": self._top_k,
        }

        if self._fetch_k is not None:
            overrides["retriever_fetch_k"] = self._fetch_k

        if question.get("as_of_date"):
            ctx = replace(ctx, as_of_date=date.fromisoformat(question["as_of_date"]))

        history = question.get("history") or []
        evidence = BenchmarkEvidence(
            schema_version=2,
            request={
                "question": question["question"],
                "history": history,
                "as_of_date": ctx.as_of_date.isoformat() if ctx.as_of_date else None,
                "user": asdict(ctx.user),
                "summary": ctx.summary,
            },
        )
        evidence_token = active_evidence.set(evidence)
        token = _settings_overrides.set(overrides)
        records = active_llm_usage.get()

        if records is None:
            records = []

        usage_token = active_llm_usage.set(records)
        started = time.monotonic()
        try:
            result = await self._rag.invoke(question=question["question"], history=history, ctx=ctx)
            elapsed = time.monotonic() - started
            evidence.completion = evidence.completion or "completed"
            evidence.answer = {
                "text": result.answer,
                "sources": result.sources,
                "completion": evidence.completion,
            }
        except BaseException as error:
            capture_failure(evidence, error)
            raise
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


def capture_failure(evidence: BenchmarkEvidence, error: BaseException) -> None:
    evidence.completion = "error" if isinstance(error, Exception) else "cancelled"

    if isinstance(error, ContextBudgetExceededError):
        evidence.completion = "prompt_budget"

    evidence.answer = {
        "text": None,
        "sources": [],
        "completion": evidence.completion,
        "error_type": type(error).__name__,
    }
    cast("EvidenceFailure", error).benchmark_evidence = evidence
