"""RAG pipeline step functions.

Each step is a pure(ish) function that takes ``RagPipelineState`` and returns
it mutated.  Steps that can terminate early (cache hit, OOD, relevance
rejection) return ``tuple[RagPipelineState, AsyncIterator[StreamEvent]]`` so
the orchestrator can yield terminal events and ``return``.

Extracted from ``RagService.stream()`` to eliminate the C901 god-method.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from domain.services.rag_policy import (
    SUFFICIENCY_ASSESSMENT_SYSTEM,
    classify_query_domain,
    classify_question_breadth,
    has_exact_reference,
    is_out_of_domain,
    should_enumerate_cases,
)
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import BREADTH_ALIASES, Breadth, LLMProvider
from domain.value_objects.search_mode import SearchMode
from domain.value_objects.stream_events import SourcesEvent, StreamEvent, TextChunk
from langchain.schema import Document as LCDocument

from config import settings
from infrastructure.bm25.hybrid import content_hash
from infrastructure.metrics.metrics import (
    RAG_BREADTH_TOTAL,
    RAG_RELEVANCE_GATE_TOTAL,
    RAG_SELF_RAG_RETRIES,
    RAG_STAGE_DURATION,
    extract_usage_from_langchain,
    record_llm_usage,
    record_rag_answer,
)
from infrastructure.ml.answer_cache import compute_question_hash, compute_visibility_scope_hash
from infrastructure.ml.clients.llm_schemas import SufficiencyAssessment
from infrastructure.ml.rag.rag_cache import check_cache, handle_cache_hit
from infrastructure.ml.rag.rag_formatting import CHARS_PER_TOKEN, format_docs
from infrastructure.ml.rag.rag_reranking import deduplicate_docs, group_by_section, rerank_documents
from infrastructure.ml.rag.rag_postprocess import (
    apply_citation_filter,
    enrich_with_neighbors,
    handle_relevance_gate,
    is_not_found_answer,
    reject_not_relevant,
    resolve_temporal_conflicts,
)
from infrastructure.ml.rag.rag_prompts import build_prompt, condense_question, decompose_question
from infrastructure.ml.rag.rag_retrieval import run_hybrid_search
from infrastructure.ml.rag.rag_sources import extract_sources
from infrastructure.vector.acl import with_domain_filter
from shared import request_id_ctx

if TYPE_CHECKING:
    from infrastructure.ml.rag_pipeline import RagPipelineState

log = logging.getLogger("default")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _domain_prompt_addendum(query_domain: str, ctx, breadth: Breadth, domain_registry) -> str | None:
    """Domain-profile prompt rules."""
    if domain_registry is None:
        return None
    try:
        profile = domain_registry.get(query_domain)
    except KeyError:
        return None
    return profile.prompt_addendum(
        breadth.value if hasattr(breadth, "value") else breadth, as_of_date=ctx.as_of_date
    )


async def _run_retrieval(
    query_for_search: str,
    fetch_k: int,
    access_filter,
    rag,
    ml_clients,
    breadth: Breadth,
    query_domain: str,
    effective_dense_weight: float,
    effective_sparse_weight: float,
) -> list[LCDocument]:
    """Run hybrid search with legal-domain fallback."""
    if query_domain == DocDomain.LEGAL.value:
        legal_filter = with_domain_filter(access_filter, DocDomain.LEGAL.value)
        candidates = await run_hybrid_search(
            query_for_search,
            fetch_k,
            legal_filter,
            rag,
            ml_clients=ml_clients,
            dense_weight=effective_dense_weight,
            sparse_weight=effective_sparse_weight,
        )
        if not candidates:
            log.info("Legal-filtered retrieval returned 0 candidates — fallback on entire corpus")
            candidates = await run_hybrid_search(
                query_for_search,
                fetch_k,
                access_filter,
                rag,
                ml_clients=ml_clients,
                dense_weight=effective_dense_weight,
                sparse_weight=effective_sparse_weight,
            )
    else:
        candidates = await run_hybrid_search(
            query_for_search,
            fetch_k,
            access_filter,
            rag,
            ml_clients=ml_clients,
            dense_weight=effective_dense_weight,
            sparse_weight=effective_sparse_weight,
        )
    return candidates


async def _apply_exact_search(
    query_for_search: str,
    candidates: list[LCDocument],
    user: dict,
    ctx,
    chunk_search,
) -> None:
    """Augment candidates with exact substring matches."""
    if chunk_search is None:
        return
    try:
        exact_results = await chunk_search.search_substring(
            query=query_for_search,
            user=user,
            group_ids=ctx.user_group_ids,
            limit=5,
            mode=SearchMode.EXACT.value,
        )
        if exact_results:
            existing_hashes = {content_hash(d.page_content) for d in candidates}
            for r in exact_results:
                h = content_hash(r.content)
                if h not in existing_hashes:
                    candidates.append(
                        LCDocument(
                            page_content=r.content,
                            metadata={
                                "source": r.filename,
                                "document_id": r.document_id,
                            },
                        )
                    )
                    existing_hashes.add(h)
            log.info("Exact-search added %d additional candidates", len(exact_results))
    except Exception as e:
        log.warning("Exact-search failed: %s", e)


async def _apply_legal_rerank_fallback(
    query_for_search: str,
    access_filter,
    rag,
    top_k: int,
    docs: list,
    ml_clients,
) -> list:
    """Fallback: rerank entire corpus for legal queries that got no docs."""
    if docs:
        return docs
    log.info("Legal query got no docs after rerank — fallback on entire corpus with rerank")
    fallback_candidates = await run_hybrid_search(
        query_for_search, rag.retriever.fetch_k, access_filter, rag, ml_clients=ml_clients
    )
    return await rerank_documents(
        query_for_search,
        fallback_candidates,
        top_n=top_k,
        reranker=ml_clients.reranker(),
        min_score=rag.rerank.min_score,
        score_gap_ratio=rag.rerank.score_gap_ratio,
    )


async def _assess_sufficiency(
    question: str, docs: list, ml_clients, llm_client=None, model: str = ""
) -> SufficiencyAssessment:
    """Self-RAG: assess whether retrieved context is sufficient to answer."""
    if not docs:
        return SufficiencyAssessment(
            is_sufficient=False,
            reasoning="No documents retrieved",
            suggested_refinement=question,
        )

    from infrastructure.llm.instructor_client import create_llm_instructor_client

    if llm_client is None:
        llm_client, model = create_llm_instructor_client()

    context = format_docs(docs, max_context_tokens=2000)
    user_msg = f"Вопрос: {question}\n\nКонтекст:\n{context}"

    async with ml_clients.llm_semaphore:
        result = await asyncio.to_thread(
            lambda: llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SUFFICIENCY_ASSESSMENT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                response_model=SufficiencyAssessment,
                max_retries=3,
            )
        )
    return result


# ── Pipeline steps ───────────────────────────────────────────────────────────


async def step_condense(state: RagPipelineState, ml_clients) -> RagPipelineState:
    """Step 1: Query condensation via LLM."""
    t0 = time.monotonic()
    if state.rag.features.condense_enabled:
        async with ml_clients.llm_semaphore:
            state.query_for_search = await condense_question(
                ml_clients.fast_llm(),
                state.question,
                state.history_messages,
            )
    else:
        state.query_for_search = state.question
    RAG_STAGE_DURATION.labels("condense").observe(time.monotonic() - t0)
    return state


async def step_check_cache(
    state: RagPipelineState,
) -> tuple[RagPipelineState, AsyncIterator[StreamEvent] | None]:
    """Step 2: Semantic answer cache lookup. Returns terminal events on hit."""
    state.vis_hash = compute_visibility_scope_hash(
        state.ctx.user_kind,
        state.ctx.user_id,
        state.ctx.user_group_ids,
    )
    state.q_hash = compute_question_hash(state.query_for_search)
    cached = await check_cache(state.rag, state.q_hash, state.vis_hash)
    if cached is not None:

        async def _cache_events():
            async for event in handle_cache_hit(cached, state.q_hash, state.t_pipeline_start):
                yield event

        state.terminal = True
        return state, _cache_events()
    return state, None


async def step_reject_ood(
    state: RagPipelineState,
) -> tuple[RagPipelineState, AsyncIterator[StreamEvent] | None]:
    """Step 3: Out-of-domain rejection. Returns terminal events if OOD."""
    if is_out_of_domain(state.query_for_search):
        log.info("Out-of-domain question rejected: %s", state.query_for_search[:100])
        RAG_RELEVANCE_GATE_TOTAL.labels(result="out_of_domain").inc()

        async def _ood_events():
            yield TextChunk(text="Информация не найдена в документах.")
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
) -> RagPipelineState:
    """Step 4: Hybrid retrieval with decomposition, exact search, reranking, enrichment."""
    ctx = state.ctx
    rag = state.rag
    query = state.query_for_search

    # ── Breadth + weights + domain ───────────────────────────────────
    raw = ctx.depth if ctx.depth in BREADTH_ALIASES else classify_question_breadth(query)
    breadth = BREADTH_ALIASES.get(raw) or Breadth(raw)
    RAG_BREADTH_TOTAL.labels(breadth=breadth).inc()

    fetch_k = rag.retriever.fetch_k_broad if breadth == Breadth.BROAD else rag.retriever.fetch_k
    rerank_top_n = rag.retriever.top_k_broad

    use_exact_ref_boost = has_exact_reference(query)
    effective_dense_weight = rag.hybrid_search.dense_weight
    effective_sparse_weight = rag.hybrid_search.sparse_weight
    if use_exact_ref_boost:
        effective_sparse_weight = rag.hybrid_search.sparse_weight * settings.exact_ref_sparse_boost

    query_domain = classify_query_domain(query)

    # ── Hybrid retrieval (with optional decomposition) ───────────────
    if rag.features.decomposition_enabled:
        t0 = time.monotonic()
        try:
            sub_queries = await decompose_question(ml_clients.fast_llm(), query)
        except Exception as e:
            log.warning("Decomposition failed, falling back to single query: %s", e)
            sub_queries = [query]

        if len(sub_queries) < 2:
            candidates = await _run_retrieval(
                query,
                fetch_k,
                state.retrieval_filter,
                rag,
                ml_clients,
                breadth,
                query_domain,
                effective_dense_weight,
                effective_sparse_weight,
            )
            RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
        else:
            retrieval_tasks = [
                _run_retrieval(
                    sq,
                    fetch_k,
                    state.retrieval_filter,
                    rag,
                    ml_clients,
                    breadth,
                    query_domain,
                    effective_dense_weight,
                    effective_sparse_weight,
                )
                for sq in sub_queries
            ]
            all_candidates_lists = await asyncio.gather(*retrieval_tasks)
            merged: list[LCDocument] = []
            for c in all_candidates_lists:
                merged.extend(c)
            candidates = deduplicate_docs(merged)

            from infrastructure.metrics.metrics import RAG_DECOMPOSED_TOTAL

            RAG_DECOMPOSED_TOTAL.inc()
            RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
            log.info(
                "Multi-query decomposition: %d sub-queries -> %d merged candidates (from %d total)",
                len(sub_queries),
                len(candidates),
                sum(len(c) for c in all_candidates_lists),
            )
    else:
        candidates = await _run_retrieval(
            query,
            fetch_k,
            state.retrieval_filter,
            rag,
            ml_clients,
            breadth,
            query_domain,
            effective_dense_weight,
            effective_sparse_weight,
        )
        sub_queries = [query]

    # ── Exact-search integration ─────────────────────────────────────
    if use_exact_ref_boost:
        await _apply_exact_search(query, candidates, state.user, ctx, chunk_search)

    # ── Detect conditional rules on FULL candidate set ───────────────
    candidate_texts = [doc.page_content for doc in candidates]
    enumerate_cases = should_enumerate_cases(query, candidate_texts)

    # ── Reranking ────────────────────────────────────────────────────
    t0 = time.monotonic()
    docs = await rerank_documents(
        query,
        candidates,
        top_n=rerank_top_n,
        reranker=ml_clients.reranker(),
        min_score=rag.rerank.min_score,
        score_gap_ratio=rag.rerank.score_gap_ratio,
    )
    RAG_STAGE_DURATION.labels("rerank").observe(time.monotonic() - t0)

    if ctx.as_of_date is not None:
        docs = resolve_temporal_conflicts(docs)

    if query_domain == "legal":
        docs = await _apply_legal_rerank_fallback(
            query,
            state.access_filter,
            rag,
            rerank_top_n,
            docs,
            ml_clients,
        )

    # ── Trim + neighbor enrichment ───────────────────────────────────
    if not enumerate_cases:
        final_top_k = rag.retriever.top_k if breadth == Breadth.NARROW else rag.retriever.top_k_broad
        docs = docs[:final_top_k]
    else:
        docs = docs[: rag.retriever.top_k_broad]

    avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0

    effective_breadth = Breadth.BROAD if enumerate_cases else breadth
    num_ctx = (
        settings.llm_num_ctx_broad if effective_breadth == Breadth.BROAD else settings.llm_num_ctx_narrow
    )
    history_chars = sum(len(m.content) for m in state.history_messages)
    question_chars = len(state.question)
    reserved_chars = history_chars + question_chars + 3000
    reserved_for_system_and_history = max(reserved_chars // CHARS_PER_TOKEN, 1500)
    max_context_tokens = max(num_ctx - reserved_for_system_and_history, 1000)

    docs = await enrich_with_neighbors(docs, enumerate_cases, chunk_search, max_context_tokens)

    # ── Write back to state ──────────────────────────────────────────
    state.breadth = breadth
    state.fetch_k = fetch_k
    state.rerank_top_n = rerank_top_n
    state.effective_dense_weight = effective_dense_weight
    state.effective_sparse_weight = effective_sparse_weight
    state.use_exact_ref_boost = use_exact_ref_boost
    state.query_domain = query_domain
    state.sub_queries = sub_queries
    state.enumerate_cases = enumerate_cases
    state.candidates = candidates
    state.docs = docs
    state.avg_sim = avg_sim
    return state


async def step_self_rag(
    state: RagPipelineState,
    ml_clients,
    chunk_search,
) -> tuple[RagPipelineState, AsyncIterator[StreamEvent] | None]:
    """Step 5: Self-RAG relevance gate with retry loop."""
    rag = state.rag
    query = state.query_for_search
    MAX_SELF_RAG_RETRIES = 1

    for attempt in range(MAX_SELF_RAG_RETRIES + 1):
        is_relevant = await handle_relevance_gate(
            query,
            state.docs,
            state.breadth,
            rag,
            ml_clients.llm_semaphore,
            ml_clients.fast_llm(),
        )
        avg_sim = sum(s for _, s in state.docs) / len(state.docs) if state.docs else 0.0

        if is_relevant:
            state.avg_sim = avg_sim
            return state, None

        if attempt < MAX_SELF_RAG_RETRIES:
            t0 = time.monotonic()
            assessment = await _assess_sufficiency(query, state.docs, ml_clients)
            RAG_STAGE_DURATION.labels("sufficiency_assess").observe(time.monotonic() - t0)

            if assessment.is_sufficient:
                log.info("Self-RAG: context deemed sufficient despite relevance gate rejection")
                state.avg_sim = avg_sim
                return state, None

            if assessment.suggested_refinement:
                log.info(
                    "Self-RAG: retrying with refined query (attempt %d/%d): %r",
                    attempt + 1,
                    MAX_SELF_RAG_RETRIES,
                    assessment.suggested_refinement,
                )
                RAG_SELF_RAG_RETRIES.inc()
                query = assessment.suggested_refinement
                state.query_for_search = query

                candidates = await _run_retrieval(
                    query,
                    state.fetch_k,
                    state.retrieval_filter,
                    rag,
                    ml_clients,
                    state.breadth,
                    state.query_domain,
                    state.effective_dense_weight,
                    state.effective_sparse_weight,
                )
                docs = await rerank_documents(
                    query,
                    candidates,
                    top_n=state.rerank_top_n,
                    reranker=ml_clients.reranker(),
                    min_score=rag.rerank.min_score,
                    score_gap_ratio=rag.rerank.score_gap_ratio,
                )
                if state.query_domain == "legal":
                    docs = await _apply_legal_rerank_fallback(
                        query,
                        state.access_filter,
                        rag,
                        state.rerank_top_n,
                        docs,
                        ml_clients,
                    )
                state.docs = docs
                continue

    avg_sim = sum(s for _, s in state.docs) / len(state.docs) if state.docs else 0.0
    state.avg_sim = avg_sim

    async def _rejection_events():
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
    domain_registry,
) -> tuple[list, list]:
    """Build prompt and format messages for LLM generation.

    Returns (messages, grouped_docs).
    """
    ctx = state.ctx
    effective_breadth = Breadth.BROAD if state.enumerate_cases else state.breadth

    domain_addendum = _domain_prompt_addendum(
        state.query_domain,
        ctx,
        effective_breadth,
        domain_registry,
    )
    prompt = build_prompt(
        effective_breadth,
        summary=ctx.summary,
        domain_addendum=domain_addendum,
        enumerate_cases=state.enumerate_cases,
    )

    history_chars = sum(len(m.content) for m in state.history_messages)
    question_chars = len(state.question)
    system_text = prompt.messages[0].content if prompt.messages else ""
    num_ctx = (
        settings.llm_num_ctx_broad if effective_breadth == Breadth.BROAD else settings.llm_num_ctx_narrow
    )
    reserved_chars = len(system_text) + history_chars + question_chars + 1000
    reserved_for_system_and_history = max(reserved_chars // CHARS_PER_TOKEN, 1500)
    max_context_tokens = max(num_ctx - reserved_for_system_and_history, 1000)

    grouped_docs = group_by_section(state.docs)
    context = format_docs(grouped_docs, max_context_tokens=max_context_tokens)
    messages = prompt.format_messages(
        context=context,
        history=state.history_messages,
        question=state.question,
    )
    return messages, grouped_docs


async def step_generate(
    state: RagPipelineState,
    ml_clients,
    messages: list,
    grouped_docs: list,
) -> AsyncIterator[StreamEvent]:
    """Stream LLM generation, yield TextChunk events.

    Caller must invoke ``step_build_context`` first and pass the resulting
    ``messages`` and ``grouped_docs``.
    """
    effective_breadth = Breadth.BROAD if state.enumerate_cases else state.breadth

    t0 = time.monotonic()
    answer_parts: list[str] = []
    last_chunk = None
    first_token_time: float | None = None
    async with ml_clients.llm_semaphore:
        async for chunk in ml_clients.llm_for_breadth(effective_breadth).astream(messages):
            last_chunk = chunk
            text = chunk.content
            if text:
                if first_token_time is None:
                    first_token_time = time.monotonic()
                answer_parts.append(text)
                yield TextChunk(text=text)
    RAG_STAGE_DURATION.labels("generate").observe(time.monotonic() - t0)

    state.full_answer = "".join(answer_parts)
    state.ttft_sec = round(first_token_time - t0, 4) if first_token_time is not None else None
    state._last_chunk = last_chunk
    state._grouped_docs = grouped_docs


def step_postprocess(state: RagPipelineState) -> RagPipelineState:
    """Step 8: PII guardrail, usage extraction, source extraction, metrics."""
    rag = state.rag
    full_answer = state.full_answer
    last_chunk = getattr(state, "_last_chunk", None)
    docs = getattr(state, "_grouped_docs", state.docs)

    # ── PII guardrail ────────────────────────────────────────────────
    if settings.pii_redaction_enabled:
        from infrastructure.ml.guardrails.guardrails import get_pii_detector

        detector = get_pii_detector()
        pii_found = detector.scan(full_answer)
        if pii_found:
            full_answer, _ = detector.scan_and_redact(full_answer)
            log.warning(
                "PII detected in LLM output [request_id=%s]: types=%s",
                request_id_ctx.get(""),
                pii_found,
            )

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
        from domain.value_objects.stream_events import UsageReport

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
        sources = extract_sources(docs, min_score=rag.source_min_score)
        sources = apply_citation_filter(rag, full_answer, sources)

    record_rag_answer(
        breadth=state.breadth.value,
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
