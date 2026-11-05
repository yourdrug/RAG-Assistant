"""Infrastructure operations for typed RAG stages."""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
import logging
import time
from typing import TYPE_CHECKING, Any, cast
import uuid

from application.rag.dto import (
    CompletedAnswer,
    ContextBudget,
    PreparedContext,
    PreparedQuery,
    RagRequest,
    RetrievalResult,
    SanitizedAnswer,
    Sufficiency,
    TerminalResponse,
)
from application.rag.stream_events import SourcesEvent, TextChunk, UsageReport
from config import get_request_settings
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.access_control import get_visibility_conditions
from domain.rag.breadth import BREADTH_ALIASES, Breadth
from domain.rag.classification import (
    classify_question_breadth,
    is_out_of_domain,
)
from domain.rag.exceptions import ContextBudgetExceededError
from domain.rag.retrieval import compute_retrieval_params
from infrastructure.domain_profile.answer_policy import classify_query_domain
from infrastructure.metrics.metrics import (
    RAG_BREADTH_TOTAL,
    RAG_CACHE_MISSES_TOTAL,
    RAG_RELEVANCE_GATE_TOTAL,
    RAG_SELF_RAG_RETRIES,
    RAG_STAGE_DURATION,
    record_llm_usage,
    record_rag_answer,
)
from infrastructure.ml.cache.answers import (
    get_corpus_revision,
)
from infrastructure.ml.cache.keys import CacheKeyBuilder, generation_identity
from infrastructure.ml.cache.pipeline import check_cache, handle_cache_hit, store_answer_cache
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.evidence_profiles.profiles import (
    EVIDENCE_PROFILES,
    active_evidence_profile,
    evidence_profile_scope,
)
from infrastructure.ml.guardrails.input_scanner import InputScanner
from infrastructure.ml.guardrails.output_scanner import OutputScanner
from infrastructure.ml.rag.benchmark_evidence import (
    active_evidence,
    capture_completion,
    capture_prepared_query,
    capture_prompt,
    capture_prompt_delivery,
)
from infrastructure.ml.rag.document_access import cache_is_accessible
from infrastructure.ml.rag.evidence_focus import format_generation_messages
from infrastructure.ml.rag.langchain_mapping import (
    chunk_value,
    lc_documents,
    lc_messages,
    message_values,
    scored_values,
)
from infrastructure.ml.rag.prompt_blocks import NOT_FOUND_ANSWER
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens
from infrastructure.ml.rag.rag_config import build_rag_settings
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection, history_to_messages
from infrastructure.ml.rag.rag_postprocess import (
    handle_relevance_gate,
    is_not_found_answer,
    reject_not_relevant,
)
from infrastructure.ml.rag.rag_prompts import (
    build_prompt,
    condense_question,
    domain_prompt_addendum,
)
from infrastructure.ml.rag.rag_relevance import assess_sufficiency, filter_cited_documents
from infrastructure.ml.rag.rag_sources import extract_sources
from infrastructure.ml.rag.retrieval_pipeline import RetrievalPipeline
from infrastructure.ml.rag.retrieval_work import RetrievalWork
from infrastructure.ml.usage_capture import extract_usage_from_langchain
from infrastructure.repositories.vector.acl import build_qdrant_filter, build_search_filter
from infrastructure.resilience.circuit_breaker import get_breaker
from shared import request_id_ctx


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from langchain_core.messages import BaseMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable

    from application.chat.ports import PIIRedactorPort
    from application.documents import DocumentAccessPort
    from application.ports.ml_clients import LLMMessagePort, MLClientPort
    from application.rag.dto import (
        SourceMetadata,
    )
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.rag.context import RagContext
    from domain.rag.ports import ChunkRetrievalRepository


MIN_HISTORY_CHARS = 3
log = logging.getLogger("default")


class RagOperations:
    def __init__(
        self,
        ml_clients: MLClientPort,
        chunk_search: ChunkRetrievalRepository | None = None,
        domain_keys: tuple[str, ...] | None = None,
        pii_redactor: PIIRedactorPort | None = None,
        *,
        document_access: DocumentAccessPort,
        domain_registry: DomainProfileRegistry | None = None,
    ) -> None:
        self.ml = ml_clients
        self.chunk_search = chunk_search
        self.domain_keys = domain_keys
        self.domain_registry = domain_registry
        self.pii_redactor = pii_redactor
        self.document_access = document_access

    @asynccontextmanager
    async def corpus(self) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            stack.enter_context(
                evidence_profile_scope(EVIDENCE_PROFILES[get_request_settings().evidence_profile])
            )

            if isinstance(self.ml, MLClientRegistry):
                await stack.enter_async_context(self.ml.corpus_operation())

            if isinstance(self.ml, MLClientRegistry):
                await stack.enter_async_context(self.ml.request_operation())

            yield

    @staticmethod
    def prepare_history_dicts(history: list[Any]) -> list[dict[str, Any]]:
        history_dicts = []

        for msg in history:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                content = msg.content

                if len(content.strip()) < MIN_HISTORY_CHARS:
                    continue

                history_dicts.append(
                    {
                        "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                        "content": content,
                    }
                )
            elif isinstance(msg, dict):
                if len(msg.get("content", "").strip()) >= MIN_HISTORY_CHARS:
                    history_dicts.append(msg)

        return history_dicts

    def request(self, question: str, history: list[Any], ctx: RagContext) -> RagRequest:
        req_id = request_id_ctx.get("")

        if not req_id:
            req_id = uuid.uuid4().hex[:12]
            request_id_ctx.set(req_id)

        return RagRequest(
            question,
            message_values(history_to_messages(self.prepare_history_dicts(history))),
            ctx,
            build_rag_settings(),
            time.monotonic(),
            req_id,
            get_request_settings().openrouter_model,
            generation_config=generation_identity(),
        )

    async def condense(self, request: RagRequest) -> PreparedQuery:
        t0 = time.monotonic()
        query = request.question

        if request.settings.features.condense_enabled:
            query = await condense_question(
                cast("Runnable[Any, BaseMessage]", self.ml.fast_llm()),
                query,
                lc_messages(request.history),
                ml_clients=self.ml,
            )

        RAG_STAGE_DURATION.labels("condense").observe(time.monotonic() - t0)
        capture_prepared_query(query)
        return PreparedQuery(request, query)

    async def scan(self, query: PreparedQuery) -> TerminalResponse | None:
        request = query.request
        scanner = InputScanner()
        verdict = scanner.scan(request.question)

        for message in request.history:
            hist = scanner.scan(message.content)

            if hist.blocked:
                log.warning("Injection detected in history message: %s", hist.reason)
                verdict = hist
                break

        if verdict.blocked:
            log.warning("Input injection blocked [request_id=%s]: %s", request.request_id, verdict.reason)
            capture_completion("input_safety")
            return self.empty_response(request)

        if verdict.flagged:
            log.info(
                "Input injection flagged (allowed) [request_id=%s]: %s", request.request_id, verdict.reason
            )

        return None

    def empty_response(self, request: RagRequest) -> TerminalResponse:
        record_rag_answer(breadth=Breadth.NARROW.value, answer="", retrieved_count=0, avg_similarity=0.0)
        RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - request.started)
        return TerminalResponse((TextChunk(text=NOT_FOUND_ANSWER), SourcesEvent(sources=[], confidence=None)))

    async def lookup(self, query: PreparedQuery) -> PreparedQuery | TerminalResponse:
        request = query.request

        if active_evidence.get() is not None or not request.settings.features.cache_enabled:
            return query

        revision = await get_corpus_revision()

        if revision is None:
            return query

        # Cache identity uses the same normalized history and request scope as retrieval.
        query = replace(query, cache_revision=revision)
        q_hash, vis_hash = CacheKeyBuilder.build(query, self.domain_keys, self.domain_registry)
        query = replace(query, q_hash=q_hash, vis_hash=vis_hash, cache_revision=revision)
        cached = await check_cache(request.settings, q_hash, vis_hash, revision)

        if cached is not None and await cache_is_accessible(
            cached, self.document_access, request.ctx.to_user_context()
        ):
            return TerminalResponse(
                tuple(
                    [
                        event
                        async for event in handle_cache_hit(
                            cached,
                            q_hash,
                            request.started,
                            rag=request.settings,
                            pii_redactor=self.pii_redactor,
                        )
                    ]
                )
            )

        if cached is not None:
            RAG_CACHE_MISSES_TOTAL.inc()

        return query

    def retrieval_work(self, prepared: PreparedQuery, result: RetrievalResult | None = None) -> RetrievalWork:
        request = prepared.request
        ctx = request.ctx
        user = ctx.to_user_context()
        scope = ctx.curator_scope
        managed = {
            "managed_client_ids": list(scope.managed_client_ids) if scope else None,
            "managed_internal_ids": list(scope.managed_internal_ids) if scope else None,
            "managed_group_ids": list(scope.managed_group_ids) if scope else None,
        }
        access_filter = build_qdrant_filter(user, list(ctx.user_group_ids), **managed)
        work = RetrievalWork(
            domain_registry=self.domain_registry,
            rag=request.settings,
            t_pipeline_start=request.started,
            question=request.question,
            ctx=ctx,
            user=user,
            access_filter=access_filter,
            retrieval_filter=build_search_filter(user, list(ctx.user_group_ids), ctx.as_of_date, **managed),
            req_id=request.request_id,
            visibility_conditions=get_visibility_conditions(
                UserKind(ctx.user_kind),
                list(ctx.user_group_ids),
                for_list=False,
                user_role=UserRole(ctx.user_role) if ctx.user_role else None,
                **managed,
            ),
            history_messages=lc_messages(request.history),
            query_for_search=prepared.query,
        )

        if result is not None:
            work.breadth = result.breadth
            work.fetch_k = result.fetch_k
            work.rerank_top_n = result.rerank_top_n
            work.effective_dense_weight = result.dense_weight
            work.effective_sparse_weight = result.sparse_weight
            work.use_exact_ref_boost = result.exact_ref_boost
            work.query_domain = result.domain
            work.docs = lc_documents(result.documents)
            work.sub_queries = list(result.sub_queries)
            work.enumerate_cases = result.enumerate_cases

        return work

    async def retrieve(
        self, query: PreparedQuery, previous: RetrievalResult | None = None
    ) -> RetrievalResult:
        work = self.retrieval_work(query, previous)

        if previous is None:
            raw = (
                work.ctx.depth
                if work.ctx.depth in BREADTH_ALIASES
                else classify_question_breadth(query.query)
            )
            work.breadth = BREADTH_ALIASES.get(raw) or Breadth(raw)
            RAG_BREADTH_TOTAL.labels(breadth=work.breadth).inc()
            params = compute_retrieval_params(work.breadth, work.rag, query.query)
            work.fetch_k = params["fetch_k"]
            work.rerank_top_n = params["rerank_top_n"]
            work.effective_dense_weight = params["effective_dense_weight"]
            work.effective_sparse_weight = params["effective_sparse_weight"]
            work.use_exact_ref_boost = params["use_exact_ref_boost"]
            work.query_domain = (
                classify_query_domain(query.query, self.domain_registry)
                if self.domain_registry is not None
                else classify_query_domain(query.query)
            )

        await RetrievalPipeline(self.ml, self.chunk_search, self.document_access).run(query.query, work)

        if work.breadth is None:
            message = "Retrieval breadth is required"
            raise ValueError(message)

        return RetrievalResult(
            query,
            work.breadth,
            work.query_domain,
            work.fetch_k,
            work.rerank_top_n,
            work.effective_dense_weight,
            work.effective_sparse_weight,
            work.use_exact_ref_boost,
            tuple(chunk_value(d) for d in work.candidates),
            scored_values(work.docs),
            tuple(work.sub_queries),
            work.enumerate_cases,
            work.avg_sim,
        )

    async def reject_ood(self, result: RetrievalResult) -> TerminalResponse | None:
        if not result.documents and is_out_of_domain(result.prepared.query):
            log.info("Out-of-domain question rejected: query_chars=%d", len(result.prepared.query))
            RAG_RELEVANCE_GATE_TOTAL.labels(result="out_of_domain").inc()
            capture_completion("out_of_domain")
            return self.empty_response(result.prepared.request)

        return None

    async def relevant(self, result: RetrievalResult) -> bool:
        return await handle_relevance_gate(
            result.prepared.query,
            lc_documents(result.documents),
            result.breadth,
            result.prepared.request.settings,
            self.ml,
        )

    async def assess(self, result: RetrievalResult) -> Sufficiency:
        t0 = time.monotonic()
        assessment = await assess_sufficiency(result.prepared.query, lc_documents(result.documents), self.ml)
        RAG_STAGE_DURATION.labels("sufficiency_assess").observe(time.monotonic() - t0)

        if assessment.is_sufficient:
            log.info("Self-RAG: context deemed sufficient despite relevance gate rejection")

        return Sufficiency(assessment.is_sufficient, assessment.suggested_refinement)

    def retry(self, refinement: str) -> None:
        log.info("Self-RAG: retrying with refined query (attempt 1/1, query_chars=%d)", len(refinement))
        RAG_SELF_RAG_RETRIES.inc()

    async def reject(self, result: RetrievalResult) -> TerminalResponse:
        capture_completion("relevance_refusal")
        return TerminalResponse(
            tuple(
                [
                    event
                    async for event in reject_not_relevant(
                        result.breadth,
                        lc_documents(result.documents),
                        result.average_similarity,
                        result.prepared.request.started,
                    )
                ]
            )
        )

    def build(
        self, result: RetrievalResult, token_counter: Callable[[list[BaseMessage]], int] | None = None
    ) -> PreparedContext:
        state = self.retrieval_work(result.prepared, result)

        if state.breadth is None:
            message = "state.breadth must be set before building context"
            raise ValueError(message)

        effective_breadth = Breadth.BROAD if state.enumerate_cases else state.breadth

        domain_addendum = domain_prompt_addendum(
            state.query_domain,
            state.ctx,
            self.domain_keys,
            self.domain_registry,
        )
        prompt = build_prompt(
            effective_breadth,
            summary=state.ctx.summary,
            domain_addendum=domain_addendum,
            enumerate_cases=state.enumerate_cases,
            question=state.question,
        )

        num_ctx = (
            state.rag.llm_num_ctx_broad
            if effective_breadth == Breadth.BROAD
            else state.rag.llm_num_ctx_narrow
        )
        num_predict = (
            state.rag.llm_num_predict_broad
            if effective_breadth == Breadth.BROAD
            else state.rag.llm_num_predict_narrow
        )
        count_tokens = token_counter or estimate_message_tokens
        history = list(state.history_messages)

        history = fit_base_history(prompt, history, state.question, num_ctx - num_predict, count_tokens)

        input_budget = num_ctx - num_predict

        while True:
            exclusions: list[dict[str, Any]] = []

            def count_context(context: str, current_history: list[BaseMessage] = history) -> int:
                return count_tokens(
                    format_generation_messages(
                        prompt, context=context, history=current_history, question=state.question
                    )
                )

            context, selected = format_docs_with_selection(
                state.docs,
                max_context_tokens=input_budget,
                context_counter=count_context,
                query_parts=state.sub_queries or [state.question],
                exclusions=exclusions,
            )

            if state.docs and not selected:
                if history:
                    history = history[2:]
                    continue

                capture_prompt(state.docs, [], context, exclusions=exclusions, status="prepared")
                raise ContextBudgetExceededError

            messages = format_generation_messages(
                prompt, context=context, history=history, question=state.question
            )

            if count_tokens(messages) > input_budget:
                raise ContextBudgetExceededError

            grounded_answer, evidence_assessment = active_evidence_profile.get().scoped_response(
                selected, context, state.question
            )
            break

        budget = ContextBudget(num_ctx, num_predict, count_tokens(messages), count_context(""))
        capture_prompt(
            state.docs,
            selected,
            context,
            exclusions=exclusions,
            status="prepared",
            messages=[{"role": message.type, "content": message.content} for message in messages],
            budget={
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "input_tokens": count_tokens(messages),
                "base_tokens": count_context(""),
                "effective_breadth": effective_breadth.value,
                "enumerate_cases": state.enumerate_cases,
                "query_parts": state.sub_queries or [state.question],
            },
        )
        return PreparedContext(
            result,
            message_values(messages),
            message_values(history),
            scored_values(selected),
            budget,
            grounded_answer,
            evidence_assessment,
        )

    async def generate(self, context: PreparedContext) -> SanitizedAnswer:
        result = context.retrieval
        request = result.prepared.request
        messages = lc_messages(context.messages)
        effective_breadth = Breadth.BROAD if result.enumerate_cases else result.breadth

        t0 = time.monotonic()
        answer_parts: list[str] = []
        last_chunk = None

        if context.grounded_answer is not None:
            capture_prompt_delivery(sent=False)
            capture_completion("grounded_response")
            answer_parts.append(context.grounded_answer)
        else:
            answer_parts, last_chunk = await self.collect_generation(messages, effective_breadth)

        RAG_STAGE_DURATION.labels("generate").observe(time.monotonic() - t0)

        capture_generation_finish(last_chunk)
        full_answer, output_verdict = OutputScanner().sanitize_output("".join(answer_parts))

        if not output_verdict.safe:
            capture_completion("output_safety")
            log.warning(
                "Output security issue [request_id=%s]: %s",
                request_id_ctx.get(""),
                output_verdict.reason,
            )

        if request.settings.pii_redaction_enabled and self.pii_redactor is not None:
            full_answer, pii_found = self.pii_redactor.scan_and_redact(full_answer)

            if pii_found:
                log.warning(
                    "PII detected in LLM output [request_id=%s]: types=%s",
                    request_id_ctx.get(""),
                    pii_found,
                )

        usage_report = None

        if last_chunk is not None:
            input_tokens, output_tokens = extract_usage_from_langchain(last_chunk)
            model_name = request.model
            record_llm_usage(
                model=model_name, operation="generate", input_tokens=input_tokens, output_tokens=output_tokens
            )
            usage_report = UsageReport(
                input_tokens=input_tokens, output_tokens=output_tokens, model=model_name, operation="generate"
            )

        return SanitizedAnswer(
            context, full_answer, usage_report, round(time.monotonic() - t0, 4) if full_answer else None
        )

    async def collect_generation(
        self, messages: list[BaseMessage], breadth: Breadth
    ) -> tuple[list[str], LLMMessagePort | None]:
        answer_parts: list[str] = []
        last_chunk: LLMMessagePort | None = None
        breaker = get_breaker("llm_generate")
        breaker.check_open()
        capture_prompt_delivery(sent=True)
        async with self.ml.generation_semaphore:
            try:
                async for chunk in self.ml.llm_for_breadth(breadth).astream(messages):
                    last_chunk = chunk
                    text = generation_chunk_text(chunk)

                    if text:
                        answer_parts.append(text)
            except Exception:
                await breaker.report_failure()
                raise
            await breaker.report_success()
        return answer_parts, last_chunk

    def finalize(self, answer: SanitizedAnswer) -> CompletedAnswer:
        context = answer.context
        result = context.retrieval
        request = result.prepared.request
        rag = request.settings
        full_answer = answer.text
        docs = lc_documents(result.documents)
        # ── Source extraction ────────────────────────────────────────────
        avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0

        if is_not_found_answer(full_answer):
            sources: list[SourceMetadata] = []
        else:
            source_docs = lc_documents(context.documents) or docs

            if rag.features.citation_filter_enabled:
                source_docs = filter_cited_documents(full_answer, source_docs)

            # Selected prompt documents must remain resolvable, even when cited
            # after score filtering would otherwise hide them.
            sources = cast("list[SourceMetadata]", extract_sources(source_docs))

        record_rag_answer(
            breadth=result.breadth.value,
            answer=full_answer,
            retrieved_count=len(docs),
            avg_similarity=avg_sim,
        )
        RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - request.started)

        return CompletedAnswer(answer, tuple(sources), min(1.0, max(0.0, avg_sim)))

    async def store(self, completed: CompletedAnswer) -> None:
        result = completed.answer.context.retrieval
        query = result.prepared

        if (
            active_evidence.get() is None
            and query.request.settings.features.cache_enabled
            and query.cache_revision is not None
        ):
            await store_answer_cache(
                lc_documents(result.documents),
                query.query,
                query.q_hash,
                query.vis_hash,
                completed.answer.text,
                sources=[dict(source) for source in completed.sources],
                expected_revision=query.cache_revision,
            )


def capture_generation_finish(last_chunk: LLMMessagePort | None) -> None:
    if last_chunk is not None:
        metadata = getattr(last_chunk, "response_metadata", None)
        reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
        capture_prompt_delivery(sent=True, finish_reason=reason if isinstance(reason, str) else None)


def fit_base_history(
    prompt: ChatPromptTemplate,
    history: list[BaseMessage],
    question: str,
    input_budget: int,
    count_tokens: Callable[[list[BaseMessage]], int],
) -> list[BaseMessage]:
    while True:
        base_messages = format_generation_messages(prompt, context="", history=history, question=question)
        max_context_tokens = input_budget - count_tokens(base_messages)

        if max_context_tokens > 0:
            return history

        if not history:
            raise ContextBudgetExceededError

        # Drop oldest complete user/assistant turns, preserving recent context.
        history = history[2:]


def generation_chunk_text(chunk: LLMMessagePort) -> str:
    if not isinstance(chunk.content, str):
        message = "Generation model returned non-text content"
        raise TypeError(message)

    return chunk.content
