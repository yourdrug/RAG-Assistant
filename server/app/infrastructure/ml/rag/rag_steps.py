"""RAG pipeline step functions.

Each step is a pure(ish) function that takes ``RagPipelineState`` and returns
it mutated.  Steps that can terminate early (cache hit, OOD, relevance
rejection) return ``tuple[RagPipelineState, AsyncIterator[StreamEvent]]`` so
the orchestrator can yield terminal events and ``return``.

Extracted from ``RagService.stream()`` to eliminate the C901 god-method.
"""

from __future__ import annotations


import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from domain.rag.retrieval import ContextBudgetExceededError


from domain.rag.classification import (
    classify_query_domain,
    classify_question_breadth,
    is_out_of_domain,
)
from domain.rag.retrieval import (
    compute_retrieval_params,
)
from domain.rag.breadth import BREADTH_ALIASES, Breadth
from domain.runtime.enums import LLMProvider
from domain.rag.prompt_blocks import NOT_FOUND_ANSWER
from domain.rag.stream_events import SourcesEvent, StreamEvent, TextChunk, UsageReport

from config import settings
from infrastructure.ml.rag.retrieval_pipeline import RetrievalPipeline
from infrastructure.ml.rag.helpers import (
    assess_sufficiency,
    domain_prompt_addendum,
)
from infrastructure.metrics.metrics import (
    RAG_BREADTH_TOTAL,
    RAG_CACHE_MISSES_TOTAL,
    RAG_RELEVANCE_GATE_TOTAL,
    RAG_SELF_RAG_RETRIES,
    RAG_STAGE_DURATION,
    extract_usage_from_langchain,
    record_llm_usage,
    record_rag_answer,
)
from infrastructure.ml.cache.keys import CacheKeyBuilder
from infrastructure.ml.cache.answers import (
    get_corpus_revision,
)
from infrastructure.ml.guardrails.input_scanner import InputScanner
from infrastructure.ml.guardrails.output_scanner import OutputScanner
from infrastructure.ml.cache.pipeline import check_cache, handle_cache_hit
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection
from infrastructure.ml.rag.rag_postprocess import (
    handle_relevance_gate,
    is_not_found_answer,
    reject_not_relevant,
)
from infrastructure.ml.rag.evidence_focus import format_generation_messages
from domain.rag.evidence.grounded_response import (
    scoped_evidence_response,
)
from infrastructure.ml.rag.rag_prompts import (
    build_prompt,
    condense_question,
)
from infrastructure.ml.rag.benchmark_evidence import capture_prompt
from infrastructure.ml.rag.rag_sources import extract_sources
from infrastructure.ml.rag.rag_relevance import filter_cited_documents
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens
from shared import request_id_ctx
from infrastructure.ml.rag.document_access import cache_is_accessible
from infrastructure.resilience.circuit_breaker import get_breaker

if TYPE_CHECKING:
    from infrastructure.ml.rag_pipeline import RagPipelineState

log = logging.getLogger("default")


# ── Pipeline steps ───────────────────────────────────────────────────────────


async def step_condense(state: RagPipelineState, ml_clients) -> RagPipelineState:
    """Step 1: Query condensation via LLM."""
    t0 = time.monotonic()
    if state.rag.features.condense_enabled:
        state.query_for_search = await condense_question(
            ml_clients.fast_llm(),
            state.question,
            state.history_messages,
            ml_clients=ml_clients,
        )
    else:
        state.query_for_search = state.question
    RAG_STAGE_DURATION.labels("condense").observe(time.monotonic() - t0)
    return state


async def step_scan_input(
    state: RagPipelineState,
) -> tuple[RagPipelineState, AsyncIterator[StreamEvent] | None]:
    """Step 1.5: Input injection scanning. Returns terminal events if blocked."""
    scanner = InputScanner()
    verdict = scanner.scan(state.question)

    # Also scan history for injection
    for msg in state.history_messages:
        if not isinstance(msg.content, str):
            continue
        hist_verdict = scanner.scan(msg.content)
        if hist_verdict.blocked:
            log.warning("Injection detected in history message: %s", hist_verdict.reason)
            verdict = hist_verdict
            break

    if verdict.blocked:
        log.warning("Input injection blocked [request_id=%s]: %s", state.req_id, verdict.reason)

        async def _injection_events():
            yield TextChunk(text=NOT_FOUND_ANSWER)
            record_rag_answer(
                breadth=Breadth.NARROW.value,
                answer="",
                retrieved_count=0,
                avg_similarity=0.0,
            )
            RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - state.t_pipeline_start)
            yield SourcesEvent(sources=[], confidence=None)

        state.terminal = True
        return state, _injection_events()

    if verdict.flagged:
        log.info("Input injection flagged (allowed) [request_id=%s]: %s", state.req_id, verdict.reason)

    return state, None


async def step_check_cache(
    state: RagPipelineState,
    domain_keys: tuple[str, ...] | None = None,
    document_access=None,
) -> tuple[RagPipelineState, AsyncIterator[StreamEvent] | None]:
    """Step 2: Semantic answer cache lookup. Returns terminal events on hit."""
    if not state.rag.features.cache_enabled:
        return state, None
    state.cache_revision = await get_corpus_revision()
    if state.cache_revision is None:
        return state, None
    state.q_hash, state.vis_hash = CacheKeyBuilder.build(state, domain_keys)
    cached = await check_cache(state.rag, state.q_hash, state.vis_hash, state.cache_revision)
    if cached is not None:
        if document_access is None or not await cache_is_accessible(cached, document_access, state.user):
            RAG_CACHE_MISSES_TOTAL.inc()
            return state, None

        async def _cache_events():
            async for event in handle_cache_hit(
                cached, state.q_hash, state.t_pipeline_start, rag=state.rag, pii_redactor=state.pii_redactor
            ):
                yield event

        state.terminal = True
        return state, _cache_events()
    return state, None


async def step_reject_ood(
    state: RagPipelineState,
) -> tuple[RagPipelineState, AsyncIterator[StreamEvent] | None]:
    """Step 3: Out-of-domain rejection. Returns terminal events if OOD."""
    if not state.docs and is_out_of_domain(state.query_for_search):
        log.info("Out-of-domain question rejected: query_chars=%d", len(state.query_for_search))
        RAG_RELEVANCE_GATE_TOTAL.labels(result="out_of_domain").inc()

        async def _ood_events():
            yield TextChunk(text=NOT_FOUND_ANSWER)
            record_rag_answer(
                breadth=Breadth.NARROW.value,
                answer="",
                retrieved_count=0,
                avg_similarity=0.0,
            )
            RAG_STAGE_DURATION.labels("total").observe(
                time.monotonic() - state.t_pipeline_start,
            )
            yield SourcesEvent(sources=[], confidence=None)

        state.terminal = True
        return state, _ood_events()
    return state, None


async def step_retrieve(
    state: RagPipelineState,
    ml_clients,
    chunk_search,
    document_access,
) -> RagPipelineState:
    """Step 4: Hybrid retrieval with decomposition, exact search, reranking, enrichment."""
    ctx = state.ctx
    rag = state.rag
    query = state.query_for_search

    # ── Breadth + weights + domain ───────────────────────────────────
    raw = ctx.depth if ctx.depth in BREADTH_ALIASES else classify_question_breadth(query)
    breadth = BREADTH_ALIASES.get(raw) or Breadth(raw)
    RAG_BREADTH_TOTAL.labels(breadth=breadth).inc()

    params = compute_retrieval_params(breadth, rag, query, exact_ref_sparse_boost=rag.exact_ref_sparse_boost)
    query_domain = classify_query_domain(query)

    state.breadth = breadth
    state.fetch_k = params["fetch_k"]
    state.rerank_top_n = params["rerank_top_n"]
    state.effective_dense_weight = params["effective_dense_weight"]
    state.effective_sparse_weight = params["effective_sparse_weight"]
    state.use_exact_ref_boost = params["use_exact_ref_boost"]
    state.query_domain = query_domain
    return await RetrievalPipeline(ml_clients, chunk_search, document_access).run(query, state)


async def step_self_rag(
    state: RagPipelineState,
    ml_clients,
    chunk_search,
    document_access,
) -> tuple[RagPipelineState, AsyncIterator[StreamEvent] | None]:
    """Step 5: Self-RAG relevance gate with retry loop."""
    rag = state.rag
    query = state.query_for_search
    MAX_SELF_RAG_RETRIES = 1

    for attempt in range(MAX_SELF_RAG_RETRIES + 1):
        if state.breadth is None:
            raise ValueError("state.breadth must be set before relevance gate")
        breadth = state.breadth
        is_relevant = await handle_relevance_gate(
            query,
            state.docs,
            breadth,
            rag,
            ml_clients,
        )
        avg_sim = sum(s for _, s in state.docs) / len(state.docs) if state.docs else 0.0

        if is_relevant:
            state.avg_sim = avg_sim
            return state, None

        if attempt < MAX_SELF_RAG_RETRIES:
            t0 = time.monotonic()
            assessment = await assess_sufficiency(query, state.docs, ml_clients)
            RAG_STAGE_DURATION.labels("sufficiency_assess").observe(time.monotonic() - t0)

            if assessment.is_sufficient:
                log.info("Self-RAG: context deemed sufficient despite relevance gate rejection")
                state.avg_sim = avg_sim
                return state, None

            if assessment.suggested_refinement:
                log.info(
                    "Self-RAG: retrying with refined query (attempt %d/%d, query_chars=%d)",
                    attempt + 1,
                    MAX_SELF_RAG_RETRIES,
                    len(assessment.suggested_refinement),
                )
                RAG_SELF_RAG_RETRIES.inc()
                query = assessment.suggested_refinement
                state.query_for_search = query

                await RetrievalPipeline(ml_clients, chunk_search, document_access).run(query, state)
                continue

    avg_sim = sum(s for _, s in state.docs) / len(state.docs) if state.docs else 0.0
    state.avg_sim = avg_sim

    async def _rejection_events():
        if state.breadth is None:
            raise ValueError("state.breadth must be set before rejection")
        async for event in reject_not_relevant(
            state.breadth,
            state.docs,
            avg_sim,
            state.t_pipeline_start,
        ):
            yield event

    state.terminal = True
    return state, _rejection_events()


def step_build_context(
    state: RagPipelineState,
    domain_keys: tuple[str, ...] | None,
    token_counter=None,
) -> tuple[list, list]:
    """Build prompt and format messages for LLM generation.

    Returns messages and retrieval candidates for downstream retrieval metrics.
    The grouped selection with citation IDs is stored in ``state._prompt_docs``.
    """
    ctx = state.ctx
    if state.breadth is None:
        raise ValueError("state.breadth must be set before building context")
    effective_breadth = Breadth.BROAD if state.enumerate_cases else state.breadth

    domain_addendum = domain_prompt_addendum(
        state.query_domain,
        ctx,
        domain_keys,
    )
    prompt = build_prompt(
        effective_breadth,
        summary=ctx.summary,
        domain_addendum=domain_addendum,
        enumerate_cases=state.enumerate_cases,
        question=state.question,
    )

    num_ctx = (
        state.rag.llm_num_ctx_broad if effective_breadth == Breadth.BROAD else state.rag.llm_num_ctx_narrow
    )
    num_predict = (
        state.rag.llm_num_predict_broad
        if effective_breadth == Breadth.BROAD
        else state.rag.llm_num_predict_narrow
    )
    count_tokens = token_counter or estimate_message_tokens
    history = list(state.history_messages)
    while True:
        base_messages = format_generation_messages(
            prompt, context="", history=history, question=state.question
        )
        max_context_tokens = num_ctx - num_predict - count_tokens(base_messages)
        if max_context_tokens > 0:
            break
        if not history:
            raise ContextBudgetExceededError()
        # Drop oldest complete user/assistant turns, preserving recent context.
        history = history[2:]
    input_budget = num_ctx - num_predict
    while True:
        exclusions: list[dict] = []

        def count_context(context, current_history=history):
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
            capture_prompt(state.docs, [], context, exclusions=exclusions)
            raise ContextBudgetExceededError()
        messages = format_generation_messages(
            prompt, context=context, history=history, question=state.question
        )
        if count_tokens(messages) > input_budget:
            raise ContextBudgetExceededError()
        state._prompt_docs = selected
        state.grounded_answer, state.evidence_assessment = scoped_evidence_response(
            selected, context, state.question
        )
        break
    state.history_messages = history
    capture_prompt(
        state.docs,
        selected,
        context,
        exclusions=exclusions,
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
    return messages, state.docs


async def step_generate(
    state: RagPipelineState,
    ml_clients,
    messages: list,
    grouped_docs: list,
) -> AsyncIterator[StreamEvent]:
    """Generate the complete answer, sanitize it, then yield a TextChunk.

    Full-answer checks must finish before any response text reaches a client.

    Caller must invoke ``step_build_context`` first and pass the resulting
    ``messages`` and ``grouped_docs``.
    """
    effective_breadth = Breadth.BROAD if state.enumerate_cases else state.breadth

    t0 = time.monotonic()
    answer_parts: list[str] = []
    last_chunk = None

    breaker = get_breaker("llm_generate")
    if state.grounded_answer is not None:
        answer_parts.append(state.grounded_answer)
    else:
        breaker.check_open()
        async with ml_clients.generation_semaphore:
            try:
                async for chunk in ml_clients.llm_for_breadth(effective_breadth).astream(messages):
                    last_chunk = chunk
                    text = chunk.content
                    if text:
                        answer_parts.append(text)
            except Exception:
                await breaker.report_failure()
                raise
            await breaker.report_success()
    RAG_STAGE_DURATION.labels("generate").observe(time.monotonic() - t0)

    full_answer, output_verdict = OutputScanner().sanitize_output("".join(answer_parts))
    if not output_verdict.safe:
        log.warning(
            "Output security issue [request_id=%s]: %s",
            request_id_ctx.get(""),
            output_verdict.reason,
        )

    if state.rag.pii_redaction_enabled and state.pii_redactor is not None:
        full_answer, pii_found = state.pii_redactor.scan_and_redact(full_answer)
        if pii_found:
            log.warning(
                "PII detected in LLM output [request_id=%s]: types=%s",
                request_id_ctx.get(""),
                pii_found,
            )

    state.full_answer = full_answer
    state.output_sanitized = True
    state.ttft_sec = round(time.monotonic() - t0, 4) if full_answer else None
    state._last_chunk = last_chunk
    state._grouped_docs = grouped_docs
    if full_answer:
        yield TextChunk(text=full_answer)


def step_postprocess(state: RagPipelineState) -> RagPipelineState:
    """Step 8: Usage, sources, and metrics for the already sanitized answer."""
    rag = state.rag
    if not state.output_sanitized:
        raise RuntimeError("Generation output must be sanitized before postprocessing")
    full_answer = state.full_answer
    last_chunk = getattr(state, "_last_chunk", None)
    docs = getattr(state, "_grouped_docs", state.docs)

    # ── Token usage extraction ───────────────────────────────────────
    usage_report = None
    if last_chunk is not None:
        input_tokens, output_tokens = extract_usage_from_langchain(last_chunk)
        model_name = (
            settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else settings.openrouter_model
        )
        record_llm_usage(
            model=model_name,
            operation="generate",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        usage_report = UsageReport(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_name,
            operation="generate",
        )

    # ── Source extraction ────────────────────────────────────────────
    avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0
    if is_not_found_answer(full_answer):
        sources: list[dict] = []
    else:
        source_docs = state._prompt_docs or docs
        if rag.features.citation_filter_enabled:
            source_docs = filter_cited_documents(full_answer, source_docs)
        # Selected prompt documents must remain resolvable, even when cited
        # after score filtering would otherwise hide them.
        sources = extract_sources(source_docs)

    record_rag_answer(
        breadth=state.breadth.value if state.breadth is not None else "",
        answer=full_answer,
        retrieved_count=len(docs),
        avg_similarity=avg_sim,
    )
    RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - state.t_pipeline_start)

    state.full_answer = full_answer
    state.sources = sources
    state.usage_report = usage_report
    state.confidence = min(1.0, max(0.0, avg_sim))
    return state
