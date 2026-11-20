"""Infrastructure operations for typed RAG stages."""

from __future__ import annotations

from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import logging
import time
from typing import TYPE_CHECKING, Any, cast
import uuid

from langchain_core.documents import Document as LCDocument

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
from config import LLMSection, config_version, request_section
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.access_control import get_visibility_conditions
from domain.rag.act_references import cites_referenced_point, referenced_act
from domain.rag.breadth import Breadth, answer_style, output_breadth, requested_breadth
from domain.rag.classification import cited_points, classify_question_breadth, is_out_of_domain
from domain.rag.exceptions import ContextBudgetExceededError
from domain.rag.outcome import AnswerOutcome
from domain.rag.ranking import RETAINED_AS_REQUESTED
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
from infrastructure.ml.cache.keys import CacheKeyBuilder, config_snapshot, generation_identity
from infrastructure.ml.cache.pipeline import check_cache, handle_cache_hit, store_answer_cache
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.evidence_profiles.profiles import (
    GenericEvidenceProfile,
    current_evidence_profile,
    evidence_profile_scope,
)
from infrastructure.ml.guardrails.input_scanner import InputScanner
from infrastructure.ml.guardrails.output_scanner import OutputScanner
from infrastructure.ml.rag.answer_stream import GuardedAnswerStream
from infrastructure.ml.rag.benchmark_evidence import (
    active_evidence,
    benchmark_mode,
    capture_completion,
    capture_gated_prompt,
    capture_prepared_query,
    capture_prompt,
    capture_prompt_delivery,
)
from infrastructure.ml.rag.chat_trace import chat_trace_scope
from infrastructure.ml.rag.document_access import cache_is_accessible
from infrastructure.ml.rag.evidence_focus import format_generation_messages
from infrastructure.ml.rag.generation_budget import prepare_generation_budget
from infrastructure.ml.rag.langchain_mapping import (
    chunk_value,
    lc_documents,
    lc_messages,
    message_values,
    scored_values,
)
from infrastructure.ml.rag.prompt_blocks import NOT_FOUND_ANSWER, is_not_found_answer
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens
from infrastructure.ml.rag.rag_config import build_rag_settings
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection, history_to_messages
from infrastructure.ml.rag.rag_postprocess import (
    TRUNCATED_ANSWER_NOTE,
    handle_relevance_gate,
    mark_partial_answer,
    reject_not_relevant,
    strip_reasoning,
)
from infrastructure.ml.rag.rag_prompts import (
    condense_question,
)
from infrastructure.ml.rag.rag_relevance import filter_cited_documents
from infrastructure.ml.rag.rag_retrieval import query_vector_scope
from infrastructure.ml.rag.rag_sources import extract_sources
from infrastructure.ml.rag.retrieval_pipeline import RetrievalPipeline
from infrastructure.ml.rag.retrieval_work import RetrievalWork
from infrastructure.ml.usage_capture import active_llm_usage, extract_usage_from_langchain
from infrastructure.repositories.vector.acl import build_qdrant_filter, build_search_filter
from infrastructure.resilience.circuit_breaker import get_breaker
from infrastructure.telemetry.tracing import step_span, traced_step, traced_stream
from shared import request_id_ctx


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    from langchain_core.messages import BaseMessage
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
    from domain.rag.settings import RagSettings
    from infrastructure.ml.evidence_profiles.profiles import EvidenceProfile
    from infrastructure.ml.guardrails.output_scanner import OutputVerdict


MIN_HISTORY_CHARS = 3
log = logging.getLogger("default")


@dataclass(frozen=True)
class ContextAssembly:
    result: RetrievalResult
    token_counter: Callable[[list[BaseMessage]], int] | None
    profile: EvidenceProfile
    prepared: PreparedContext
    docs: tuple[Any, ...]
    selected: tuple[Any, ...]
    context: str
    exclusions: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]
    budget: dict[str, Any]


# Per-request (per-task) memo of the last packed context; never shared across requests.
context_assembly_memo: ContextVar[ContextAssembly | None] = ContextVar("context_assembly_memo", default=None)


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
        evidence_profile: EvidenceProfile | None = None,
    ) -> None:
        self.ml = ml_clients
        # The domain pack's evidence rules; neutral rules when none is supplied.
        self.evidence_profile = evidence_profile or GenericEvidenceProfile()
        self.chunk_search = chunk_search
        self.domain_keys = domain_keys
        self.domain_registry = domain_registry
        self.pii_redactor = pii_redactor
        self.document_access = document_access

    @asynccontextmanager
    async def corpus(self) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            stack.enter_context(step_span("rag.pipeline"))
            stack.enter_context(evidence_profile_scope(self.evidence_profile))
            stack.enter_context(query_vector_scope())
            stack.enter_context(chat_trace_scope(self.pii_redactor))

            if isinstance(self.ml, MLClientRegistry):
                await stack.enter_async_context(self.ml.corpus_operation())

            if isinstance(self.ml, MLClientRegistry):
                await stack.enter_async_context(self.ml.request_operation())

            token = active_llm_usage.set([]) if active_llm_usage.get() is None else None
            try:
                yield
            finally:
                if token is not None:
                    active_llm_usage.reset(token)

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

        settings = build_rag_settings()
        config_hash, config_data = config_snapshot(
            settings,
            {
                "id": ctx.space.id,
                "pack_key": ctx.space.pack_key,
            }
            if ctx.space is not None
            else None,
            self.domain_registry,
        )
        records = active_llm_usage.get()
        evidence = active_evidence.get()
        return RagRequest(
            question,
            message_values(history_to_messages(self.prepare_history_dicts(history))),
            ctx,
            settings,
            time.monotonic(),
            req_id,
            request_section(LLMSection).llm_model,
            generation_config=generation_identity(),
            config_hash=config_hash,
            config_version=config_version(),
            config_data=config_data,
            evidence_profile=config_data["evidence_profile"],
            usage_records=records if records is not None else [],
            usage_start=len(records) if records is not None else 0,
            trace_data=evidence.trace_data if evidence is not None and not evidence.benchmark_mode else None,
        )

    @traced_step("rag.condense")
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

    @traced_step("rag.input_safety")
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
            capture_completion(AnswerOutcome.INPUT_BLOCKED)
            return replace(self.empty_response(request), outcome=AnswerOutcome.INPUT_BLOCKED)

        if verdict.flagged:
            log.info(
                "Input injection flagged (allowed) [request_id=%s]: %s", request.request_id, verdict.reason
            )

        return None

    def empty_response(self, request: RagRequest) -> TerminalResponse:
        record_rag_answer(breadth=Breadth.NARROW.value, answer="", retrieved_count=0, avg_similarity=0.0)
        RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - request.started)
        return TerminalResponse((TextChunk(text=NOT_FOUND_ANSWER), SourcesEvent(sources=[], confidence=None)))

    @traced_step("rag.cache_lookup")
    async def lookup(self, query: PreparedQuery) -> PreparedQuery | TerminalResponse:
        request = query.request

        if benchmark_mode() or not request.settings.features.cache_enabled:
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
                ),
                outcome=AnswerOutcome.CACHE_HIT,
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
            domain_keys=self.domain_keys,
            rag=request.settings,
            t_pipeline_start=request.started,
            question=request.question,
            ctx=ctx,
            user=user,
            access_filter=access_filter,
            retrieval_filter=build_search_filter(
                user,
                list(ctx.user_group_ids),
                current_evidence_profile().chunk_selection(ctx.as_of_date),
                space_id=ctx.space.id if ctx.space is not None else None,
                **managed,
            ),
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
            work.candidates = [
                LCDocument(page_content=chunk.content, metadata=dict(chunk.metadata))
                for chunk in result.candidates
            ]
            work.evidence_query = result.prepared.query
            work.sub_queries = list(result.sub_queries)
            work.enumerate_cases = result.enumerate_cases

        return work

    @traced_step("rag.retrieve")
    async def retrieve(
        self, query: PreparedQuery, previous: RetrievalResult | None = None
    ) -> RetrievalResult:
        work = self.retrieval_work(query, previous)

        if previous is None:
            work.breadth = requested_breadth(work.ctx.depth) or Breadth(
                classify_question_breadth(query.query)
            )
            RAG_BREADTH_TOTAL.labels(breadth=work.breadth).inc()
            params = compute_retrieval_params(work.rag, query.query)
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
            previous.prepared if previous is not None else query,
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
            (work.retrieval_ms or 0) + (previous.retrieval_ms or 0 if previous else 0),
            (work.rerank_ms or 0) + (previous.rerank_ms or 0 if previous else 0),
            query.query if previous is not None else None,
            work.top_rerank_score,
        )

    @traced_step("rag.reject_ood")
    async def reject_ood(self, result: RetrievalResult) -> TerminalResponse | None:
        if not result.documents and is_out_of_domain(result.prepared.query):
            log.info("Out-of-domain question rejected: query_chars=%d", len(result.prepared.query))
            RAG_RELEVANCE_GATE_TOTAL.labels(result="out_of_domain").inc()
            capture_completion(AnswerOutcome.OUT_OF_DOMAIN)
            return replace(self.empty_response(result.prepared.request), outcome=AnswerOutcome.OUT_OF_DOMAIN)

        return None

    @traced_step("rag.evidence_gate")
    async def gate(self, result: RetrievalResult) -> Sufficiency:
        request = result.prepared.request
        assembly = (
            self.assemble(result, None, capture=False)
            if request.settings.features.relevance_gate_enabled
            else None
        )
        relevant, refinement = await handle_relevance_gate(
            request.question,
            lc_documents(result.documents),
            request.settings,
            self.ml,
            context=assembly.prepared.evidence_context if assembly is not None else None,
            history=request.history,
        )

        if not relevant and assembly is not None:
            capture_gated_prompt(assembly.selected, assembly.context, list(assembly.exclusions))

        return Sufficiency(relevant, refinement or None)

    def retry(self, refinement: str) -> None:
        log.info("Self-RAG: retrying with refined query (attempt 1/1, query_chars=%d)", len(refinement))
        RAG_SELF_RAG_RETRIES.inc()

    @traced_step("rag.reject")
    async def reject(self, result: RetrievalResult) -> TerminalResponse:
        capture_completion(AnswerOutcome.NOT_FOUND)
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
        self,
        result: RetrievalResult,
        token_counter: Callable[[list[BaseMessage]], int] | None = None,
        *,
        capture: bool = True,
    ) -> PreparedContext:
        assembly = self.assemble(result, token_counter, capture=capture)
        capture_prompt(
            assembly.docs,
            assembly.selected,
            assembly.context,
            exclusions=list(assembly.exclusions),
            status="prepared" if capture else None,
            messages=list(assembly.messages),
            budget=assembly.budget,
        )
        return assembly.prepared

    def assemble(
        self,
        result: RetrievalResult,
        token_counter: Callable[[list[BaseMessage]], int] | None,
        *,
        capture: bool,
    ) -> ContextAssembly:
        # relevant(), assess() and the final build() pack the same result into the
        # token budget; reuse the packing while the inputs are the same objects.
        profile = current_evidence_profile()
        cached = context_assembly_memo.get()

        if (
            cached is not None
            and cached.result is result
            and cached.token_counter is token_counter
            and cached.profile is profile
        ):
            return cached

        assembly = self.pack_context(result, token_counter, profile, capture=capture)
        context_assembly_memo.set(assembly)
        return assembly

    def pack_context(
        self,
        result: RetrievalResult,
        token_counter: Callable[[list[BaseMessage]], int] | None,
        profile: EvidenceProfile,
        *,
        capture: bool,
    ) -> ContextAssembly:
        state = self.retrieval_work(result.prepared, result)

        if state.breadth is None:
            message = "state.breadth must be set before building context"
            raise ValueError(message)

        style = answer_style(state.ctx.depth)

        count_tokens = token_counter or estimate_message_tokens
        generation = prepare_generation_budget(state, count_tokens)
        prompt, history = generation.prompt, generation.history
        num_ctx, num_predict, input_budget = (
            generation.num_ctx,
            generation.num_predict,
            generation.input_budget,
        )

        while True:
            exclusions: list[dict[str, Any]] = []

            # Evidence is selected against the widest style prompt and the largest output reserve,
            # so switching the answer style can never add or drop a source.
            def count_context(context: str, current_history: list[BaseMessage] = history) -> int:
                return count_tokens(
                    format_generation_messages(
                        generation.selection_prompt,
                        context=context,
                        history=current_history,
                        question=state.question,
                    )
                )

            context, selected = format_docs_with_selection(
                state.docs,
                max_context_tokens=input_budget,
                context_counter=count_context,
                query_parts=state.sub_queries or [state.question],
                exclusions=exclusions,
                preferred=packing_preference(state.question, profile),
                named=named_act_documents(state.question, profile),
            )

            if state.docs and not selected:
                if history:
                    history = history[2:]
                    continue

                capture_prompt(
                    state.docs, [], context, exclusions=exclusions, status="prepared" if capture else None
                )
                raise ContextBudgetExceededError

            messages = format_generation_messages(
                prompt, context=context, history=history, question=state.question
            )
            input_tokens = count_tokens(messages)

            if input_tokens > input_budget:
                raise ContextBudgetExceededError

            grounded_answer, evidence_assessment = profile.scoped_response(selected, context, state.question)
            break

        # Diagnostics describe the prompt actually sent; the widest prompt is only for selection.
        def count_sent(context: str, current_history: list[BaseMessage] = history) -> int:
            return count_tokens(
                format_generation_messages(
                    prompt, context=context, history=current_history, question=state.question
                )
            )

        base_tokens = count_sent("")
        budget = ContextBudget(num_ctx, num_predict, input_tokens, base_tokens)
        prepared = PreparedContext(
            result,
            message_values(messages),
            message_values(history),
            scored_values(selected),
            budget,
            grounded_answer,
            evidence_assessment,
            context,
        )
        trace_budget = production_prompt_budget(base_tokens, input_tokens, exclusions, count_sent)
        return ContextAssembly(
            result=result,
            token_counter=token_counter,
            profile=profile,
            prepared=prepared,
            docs=tuple(state.docs),
            selected=tuple(selected),
            context=context,
            exclusions=tuple(exclusions),
            messages=tuple({"role": message.type, "content": message.content} for message in messages),
            budget={
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "input_tokens": input_tokens,
                "base_tokens": base_tokens,
                "answer_style": style.value,
                "input_budget": input_budget,
                "enumerate_cases": state.enumerate_cases,
                "query_parts": state.sub_queries or [state.question],
                **trace_budget,
            },
        )

    def sanitize_answer(self, text: str, rag: RagSettings) -> tuple[str, OutputVerdict, list[str]]:
        """Apply the output scanner, partial-answer marking and PII redaction."""
        answer, verdict = OutputScanner().sanitize_output(strip_reasoning(text))
        answer = mark_partial_answer(answer)
        pii_found: list[str] = []

        if rag.pii_redaction_enabled and self.pii_redactor is not None:
            answer, pii_found = self.pii_redactor.scan_and_redact(answer)

        return answer, verdict, pii_found

    @traced_stream("rag.generate")
    async def stream_answer(self, context: PreparedContext) -> AsyncGenerator[TextChunk | SanitizedAnswer]:
        """Yield sanitized answer text as it is generated, then the final answer.

        Released chunks form a prefix of the final answer when the guardrails
        allow it; otherwise streaming stops and only the final answer is
        authoritative.
        """
        run = GenerationRun(started=time.monotonic())

        if context.grounded_answer is None:
            async with aclosing(self.released_text(context, run)) as released:
                async for text in released:
                    yield TextChunk(text=text)
        else:
            capture_prompt_delivery(sent=False)
            run.parts.append(context.grounded_answer)

        RAG_STAGE_DURATION.labels("generate").observe(time.monotonic() - run.started)
        answer = self.sanitized(context, run)
        rest = run.guard.finish(answer.text) if run.guard is not None else answer.text

        if rest:
            yield TextChunk(text=rest)

        yield answer

    async def released_text(self, context: PreparedContext, run: GenerationRun) -> AsyncGenerator[str]:
        result = context.retrieval
        settings = result.prepared.request.settings
        breadth = output_breadth(answer_style(result.prepared.request.ctx.depth))
        run.guard = GuardedAnswerStream(lambda text: self.stream_check(text, settings))

        async with aclosing(self.generation_chunks(lc_messages(context.messages), breadth)) as chunks:
            async for chunk in chunks:
                run.last_chunk = chunk
                # With usage streaming the final chunk carries only usage, not the finish reason.
                run.finish_reason = chunk_finish_reason(chunk) or run.finish_reason
                text = generation_chunk_text(chunk)

                if not text:
                    continue

                if run.first_token_at is None:
                    run.first_token_at = time.monotonic()

                run.parts.append(text)
                released = run.guard.push(text)

                if released:
                    yield released

    def sanitized(self, context: PreparedContext, run: GenerationRun) -> SanitizedAnswer:
        request = context.retrieval.prepared.request
        finish_reason = run.finish_reason

        if run.last_chunk is not None:
            capture_prompt_delivery(sent=True, finish_reason=finish_reason)

        full_answer, output_verdict, pii_found = self.sanitize_answer("".join(run.parts), request.settings)
        truncated = finish_reason == "length"

        if truncated:
            log.warning(
                "LLM answer hit the output token limit [request_id=%s]: chars=%d",
                request_id_ctx.get(""),
                len(full_answer),
            )

            if output_verdict.safe and full_answer:
                full_answer += TRUNCATED_ANSWER_NOTE

        if not output_verdict.safe:
            log.warning(
                "Output security issue [request_id=%s]: %s",
                request_id_ctx.get(""),
                output_verdict.reason,
            )

        if pii_found:
            log.warning(
                "PII detected in LLM output [request_id=%s]: types=%s",
                request_id_ctx.get(""),
                pii_found,
            )

        outcome = AnswerOutcome.GROUNDED if context.grounded_answer is not None else AnswerOutcome.GENERATED

        if not output_verdict.safe:
            outcome = AnswerOutcome.OUTPUT_BLOCKED

        capture_completion(outcome)
        return SanitizedAnswer(
            context,
            full_answer,
            generation_usage(request.model, run.last_chunk),
            round(time.monotonic() - run.started, 4) if full_answer else None,
            round(run.first_token_at - run.started, 4) if run.first_token_at is not None else None,
            outcome,
            truncated=truncated,
            finish_reason=finish_reason,
        )

    def stream_check(self, text: str, rag: RagSettings) -> tuple[str, bool]:
        answer, verdict, _ = self.sanitize_answer(text, rag)
        return answer, verdict.safe

    async def generation_chunks(
        self, messages: list[BaseMessage], breadth: Breadth
    ) -> AsyncGenerator[LLMMessagePort]:
        """Stream raw LLM chunks under the generation semaphore and circuit breaker."""
        breaker = get_breaker("llm_generate")
        breaker.check_open()
        capture_prompt_delivery(sent=True)
        async with self.ml.generation_semaphore:
            try:
                stream = self.ml.llm_for_breadth(breadth).astream(messages)
                try:
                    async for chunk in stream:
                        yield chunk
                finally:
                    # A closed consumer must release the provider stream and its client lease now.
                    aclose = getattr(stream, "aclose", None)

                    if aclose is not None:
                        await aclose()
            except Exception:
                await breaker.report_failure()
                raise
            await breaker.report_success()

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

    @traced_step("rag.cache_store")
    async def store(self, completed: CompletedAnswer) -> None:
        result = completed.answer.context.retrieval
        query = result.prepared

        # A cut-off answer is a provider budget failure, not an answer to reuse.
        if (
            not benchmark_mode()
            and not completed.answer.truncated
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
                answer_metadata={
                    "breadth": result.breadth.value,
                    "domain": result.domain,
                    "confidence": completed.confidence,
                },
            )


@dataclass
class GenerationRun:
    """Mutable state of one answer generation, shared by its streaming steps."""

    started: float
    parts: list[str] = field(default_factory=list)
    last_chunk: LLMMessagePort | None = None
    first_token_at: float | None = None
    guard: GuardedAnswerStream | None = None
    finish_reason: str | None = None


def generation_usage(model: str, last_chunk: LLMMessagePort | None) -> UsageReport | None:
    if last_chunk is None:
        return None

    input_tokens, output_tokens = extract_usage_from_langchain(last_chunk)
    record_llm_usage(
        model=model, operation="generate", input_tokens=input_tokens, output_tokens=output_tokens
    )
    return UsageReport(
        input_tokens=input_tokens, output_tokens=output_tokens, model=model, operation="generate"
    )


def chunk_finish_reason(chunk: LLMMessagePort) -> str | None:
    metadata = getattr(chunk, "response_metadata", None)
    reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
    return reason if isinstance(reason, str) else None


def generation_chunk_text(chunk: LLMMessagePort) -> str:
    if not isinstance(chunk.content, str):
        message = "Generation model returned non-text content"
        raise TypeError(message)

    return chunk.content


def production_prompt_budget(
    base_tokens: int,
    input_tokens: int,
    exclusions: list[dict[str, Any]],
    count_context: Callable[[str, list[BaseMessage]], int],
) -> dict[str, int]:
    evidence = active_evidence.get()

    if evidence is None or evidence.benchmark_mode:
        return {}

    tokens_without_history = count_context("", [])
    return {
        "base_tokens": tokens_without_history,
        "history_tokens": base_tokens - tokens_without_history,
        "context_tokens": input_tokens - base_tokens,
        "documents_not_fitted": sum(item["reason"] in {"budget", "oversized_block"} for item in exclusions),
    }


def cited_point_preference(
    question: str, profile: EvidenceProfile
) -> Callable[[tuple[LCDocument, float]], bool] | None:
    """Context packing tries first the chunks holding a point the question cites by number."""
    points = cited_points(question)

    if not points:
        return None

    acts = profile.named_acts(question)
    return lambda item: cites_referenced_point(item[0].metadata, points, acts)


def packing_preference(question: str, profile: EvidenceProfile) -> Callable[[tuple[LCDocument, float]], bool]:
    """Packing tries first a cited point and the provisions reranking kept past its window as requested."""
    cited = cited_point_preference(question, profile)
    return lambda item: (
        bool(item[0].metadata.get(RETAINED_AS_REQUESTED)) or (cited is not None and cited(item))
    )


def named_act_documents(
    question: str, profile: EvidenceProfile
) -> Callable[[tuple[LCDocument, float]], bool] | None:
    """Chunks of an act the question names may fill the context beyond one document's share."""
    acts = profile.named_acts(question)

    if not acts:
        return None

    return lambda item: referenced_act(item[0].metadata, acts) is not None
