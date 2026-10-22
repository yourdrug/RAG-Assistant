"""Order RAG stages, bounded retry, and public stream events."""

from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import replace
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
from domain.knowledge.enums import DocDomain
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.result import RagResult
from domain.rag.retrieval import compute_reranker_score
from domain.rag.stream_events import PipelineMetaEvent, SourcesEvent, StatusEvent, StreamEvent, TextChunk


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
        self, question: str, history: list[object], ctx: RagContext
    ) -> AsyncGenerator[StreamEvent]:
        async with self.corpus():
            request = self.preparation.request(question, history, ctx)
            async with aclosing(self.events(request)) as events:
                async for event in events:
                    yield event

    async def events(self, request: RagRequest) -> AsyncGenerator[StreamEvent]:
        query = await self.prepare(request)

        if isinstance(query, TerminalResponse):
            for event in query.events:
                yield event

            return

        yield StatusEvent(stage="searching")
        result = await self.retrieval.retrieve(query)
        yield StatusEvent(stage="reranking")
        approved = await self.approve(result)

        if isinstance(approved, TerminalResponse):
            for event in approved.events:
                yield event

            return

        result = approved
        yield StatusEvent(stage="generating")
        context = self.context.build(result)
        answer = await self.generation.generate(context)

        if answer.text:
            yield TextChunk(text=answer.text)

        completed = self.finalization.finalize(answer)
        await self.cache.store(completed)
        yield SourcesEvent(
            sources=[dict(source) for source in completed.sources],
            confidence=completed.confidence,
            usage=answer.usage,
        )
        yield PipelineMetaEvent(breadth=result.breadth, domain=result.domain, ttft_sec=answer.ttft_sec)

    async def prepare(self, request: RagRequest) -> PreparedQuery | TerminalResponse:
        query = await self.preparation.condense(request)
        terminal = await self.safety.scan(query)

        if terminal is not None:
            return terminal

        return await self.cache.lookup(query)

    async def approve(self, result: RetrievalResult) -> RetrievalResult | TerminalResponse:
        terminal = await self.evidence.reject_ood(result)

        if terminal is not None:
            return terminal

        for attempt in range(2):
            if await self.evidence.relevant(result):
                return result

            if attempt == 0:
                assessment = await self.evidence.assess(result)

                if assessment.sufficient:
                    return result

                if assessment.refinement:
                    self.evidence.retry(assessment.refinement)
                    query = replace(result.prepared, query=assessment.refinement)
                    result = await self.retrieval.retrieve(query, previous=result)

        return await self.evidence.reject(result)

    async def invoke(
        self,
        question: str,
        history: list[object],
        ctx: RagContext,
    ) -> RagResult:
        async with self.corpus():
            request = self.preparation.request(question, history, ctx)
            return await self.result(request)

    async def result(self, request: RagRequest) -> RagResult:
        answer_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        breadth = Breadth.NARROW
        domain = DocDomain.GENERAL.value
        retrieval_count = 0
        reranker_score: float | None = None
        confidence: float | None = None
        usage_report = None
        ttft_sec: float | None = None

        async for event in self.events(request):
            if isinstance(event, SourcesEvent):
                sources = event.sources
                confidence = event.confidence
                usage_report = event.usage
            elif isinstance(event, PipelineMetaEvent):
                breadth = event.breadth or breadth

                domain = event.domain or domain

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
            model_used=request.model,
            input_tokens=usage_report.input_tokens if usage_report else None,
            output_tokens=usage_report.output_tokens if usage_report else None,
            ttft_sec=ttft_sec,
        )
