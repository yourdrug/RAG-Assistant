"""Order RAG stages, bounded retry, and public stream events."""

from collections.abc import AsyncGenerator, Iterator, Sequence
from contextlib import aclosing
from dataclasses import fields, replace
import time
from typing import Any

from application.rag.dto import PreparedQuery, RagRequest, RetrievalResult, TerminalResponse
from application.rag.ports import (
    AnswerCache,
    ContextPreparation,
    CorpusLease,
    EvidenceGate,
    Finalization,
    Generation,
    InputSafety,
    RequestPreparation,
    Retrieval,
)
from application.rag.stream_events import PipelineMetaEvent, SourcesEvent, StatusEvent, StreamEvent, TextChunk
from domain.knowledge.enums import DocDomain, DomainKey
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.metadata import AnswerMetadata, TokenUsage
from domain.rag.outcome import AnswerOutcome
from domain.rag.result import RagResult


class RagService:
    def __init__(  # noqa: PLR0913 - the workflow receives independent stage ports.
        self,
        *,
        preparation: RequestPreparation,
        safety: InputSafety,
        cache: AnswerCache,
        retrieval: Retrieval,
        evidence: EvidenceGate,
        context: ContextPreparation,
        generation: Generation,
        finalization: Finalization,
        corpus: CorpusLease,
    ) -> None:
        self.preparation = preparation
        self.safety = safety
        self.cache = cache
        self.retrieval = retrieval
        self.evidence = evidence
        self.context = context
        self.generation = generation
        self.finalization = finalization
        self.corpus = corpus

    async def stream(
        self, question: str, history: Sequence[object], ctx: RagContext
    ) -> AsyncGenerator[StreamEvent]:
        async with self.corpus():
            request = self.preparation.request(question, list(history), ctx)
            async with aclosing(self.events(request)) as events:
                async for event in events:
                    yield event

    async def events(self, request: RagRequest) -> AsyncGenerator[StreamEvent]:
        meta = PipelineMetaEvent(
            trace_data=request.trace_data,
            request_id=request.request_id,
            model_used=request.model,
            config_hash=request.config_hash,
            config_data=request.config_data,
            config_version=request.config_version,
            evidence_profile=request.evidence_profile,
        )
        yield meta
        query = await self.prepare(request)

        if isinstance(query, TerminalResponse):
            for event in terminal_events(query, meta, request):
                yield event

            return

        meta = replace(meta, condensed_query=query.query)
        yield meta
        yield StatusEvent(stage="searching")
        result = await self.retrieval.retrieve(query)
        yield StatusEvent(stage="reranking")
        meta = retrieval_metadata(meta, result)
        yield meta
        approved = await self.approve(result)

        if isinstance(approved, TerminalResponse):
            for event in terminal_events(approved, meta, request):
                yield event

            return

        result = approved
        meta = retrieval_metadata(meta, result)
        yield meta
        yield StatusEvent(stage="generating")
        context = self.context.build(result)
        answer = await self.generation.generate(context)

        ttft_sec = (time.monotonic() - request.started) if answer.text else None

        if answer.text:
            yield TextChunk(text=answer.text)

        completed = self.finalization.finalize(answer)
        await self.cache.store(completed)
        yield SourcesEvent(
            sources=[dict(source) for source in completed.sources],
            confidence=completed.confidence,
            usage=answer.usage,
        )

        if request.trace_data is not None:
            request.trace_data["outcome"] = answer.outcome

        yield replace(
            meta,
            outcome=answer.outcome,
            generation_sec=answer.generation_sec,
            ttft_sec=ttft_sec,
            aux_tokens=auxiliary_tokens(request),
            aux_usage=auxiliary_usage(request),
        )

    async def prepare(self, request: RagRequest) -> PreparedQuery | TerminalResponse:
        query = await self.preparation.condense(request)
        terminal = await self.safety.scan(query)

        if terminal is not None:
            return replace(
                terminal, events=(PipelineMetaEvent(condensed_query=query.query), *terminal.events)
            )

        cached = await self.cache.lookup(query)

        if isinstance(cached, TerminalResponse):
            return replace(cached, events=(PipelineMetaEvent(condensed_query=query.query), *cached.events))

        return cached

    async def approve(self, result: RetrievalResult) -> RetrievalResult | TerminalResponse:
        terminal = await self.evidence.reject_ood(result)

        if terminal is not None:
            return replace(terminal, retrieval=result)

        for attempt in range(2):
            if await self.evidence.relevant(result):
                return result

            if attempt == 0:
                assessment = await self.evidence.assess(result)

                if assessment.sufficient:
                    return result

                if not assessment.refinement:
                    # Re-gating the unchanged evidence would only repeat the rejection.
                    break

                self.evidence.retry(assessment.refinement)
                query = replace(result.prepared, query=assessment.refinement)
                result = await self.retrieval.retrieve(query, previous=result)

        return replace(await self.evidence.reject(result), retrieval=result)

    async def invoke(
        self,
        question: str,
        history: Sequence[object],
        ctx: RagContext,
    ) -> RagResult:
        async with self.corpus():
            request = self.preparation.request(question, list(history), ctx)
            return await self.result(request)

    async def result(self, request: RagRequest) -> RagResult:
        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        breadth = Breadth.NARROW
        domain = DocDomain.GENERAL.value
        confidence: float | None = None
        usage_report = None
        meta = PipelineMetaEvent()

        async for event in self.events(request):
            if isinstance(event, SourcesEvent):
                sources = event.sources
                confidence = event.confidence
                usage_report = event.usage
            elif isinstance(event, PipelineMetaEvent):
                meta = event
            elif isinstance(event, TextChunk):
                answer_parts.append(event.text)

        return RagResult(
            answer="".join(answer_parts),
            sources=sources,
            breadth=meta.breadth or breadth,
            domain=DomainKey(meta.domain or domain),
            sources_count=len(sources),
            confidence=confidence,
            model_used=meta.model_used,
            input_tokens=usage_report.input_tokens if usage_report else None,
            output_tokens=usage_report.output_tokens if usage_report else None,
            generation_sec=meta.generation_sec,
            ttft_sec=meta.ttft_sec,
            config_data=meta.config_data,
            aux_usage=meta.aux_usage,
            **{field.name: getattr(meta, field.name) for field in fields(AnswerMetadata)},
        )


def auxiliary_usage(request: RagRequest) -> TokenUsage:
    records = [
        record
        for record in request.usage_records[request.usage_start :]
        if record.get("operation") not in {"generate", "embedding", "reranking"}
    ]
    return TokenUsage(
        input_tokens=sum(int(record["input_tokens"]) for record in records)
        if all(record.get("input_tokens") is not None for record in records)
        else None,
        output_tokens=sum(int(record["output_tokens"]) for record in records)
        if all(record.get("output_tokens") is not None for record in records)
        else None,
    )


def auxiliary_tokens(request: RagRequest) -> int | None:
    usage = auxiliary_usage(request)

    if usage.input_tokens is None or usage.output_tokens is None:
        return None

    return usage.input_tokens + usage.output_tokens


def retrieval_metadata(meta: PipelineMetaEvent, result: RetrievalResult) -> PipelineMetaEvent:
    return replace(
        meta,
        breadth=result.breadth,
        domain=result.domain,
        candidates_count=len(result.candidates),
        top_rerank_score=result.top_rerank_score,
        condensed_query=result.prepared.query,
        sub_queries=list(result.sub_queries),
        refined_query=result.refined_query,
        retrieval_ms=result.retrieval_ms,
        rerank_ms=result.rerank_ms,
    )


def terminal_events(
    terminal: TerminalResponse, meta: PipelineMetaEvent, request: RagRequest
) -> Iterator[StreamEvent]:
    if terminal.retrieval is not None:
        meta = retrieval_metadata(meta, terminal.retrieval)

    outcome = terminal.outcome

    for event in terminal.events:
        if isinstance(event, PipelineMetaEvent):
            meta = replace(
                meta,
                breadth=event.breadth or meta.breadth,
                domain=event.domain or meta.domain,
                condensed_query=event.condensed_query or meta.condensed_query,
                generation_sec=event.generation_sec,
            )

            if terminal.outcome == AnswerOutcome.CACHE_HIT:
                outcome = event.outcome
        else:
            yield event

    if request.trace_data is not None:
        request.trace_data["outcome"] = outcome

    yield replace(
        meta, outcome=outcome, aux_tokens=auxiliary_tokens(request), aux_usage=auxiliary_usage(request)
    )
