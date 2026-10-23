"""Retrieval orchestration: search, reranking, authorized context enrichment."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from langchain.schema import Document as LCDocument

from domain.knowledge.enums import DocDomain, SearchMode
from domain.rag.classification import should_enumerate_cases
from domain.rag.evidence.queries import evidence_search_queries
from domain.rag.retrieval import compute_context_budget, select_final_top_k
from domain.shared.hashing import content_hash
from infrastructure.metrics.metrics import RAG_DECOMPOSED_TOTAL, RAG_STAGE_DURATION
from infrastructure.ml.rag.applicable_context import enrich_applicable_context
from infrastructure.ml.rag.benchmark_evidence import active_evidence, capture_exclusions, capture_retrieval
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata
from infrastructure.ml.rag.context_selection import ExclusionReason, exclusion_record
from infrastructure.ml.rag.document_access import filter_documents, filter_scored_documents
from infrastructure.ml.rag.field_evidence import augment_field_candidates
from infrastructure.ml.rag.linked_context import enrich_linked_context
from infrastructure.ml.rag.rag_postprocess import enrich_with_neighbors, resolve_temporal_conflicts
from infrastructure.ml.rag.rag_prompts import decompose_question
from infrastructure.ml.rag.rag_reranking import deduplicate_docs, filter_scope_candidates, rerank_documents
from infrastructure.ml.rag.rag_retrieval import run_hybrid_search
from infrastructure.repositories.vector.acl import with_domain_filter


if TYPE_CHECKING:
    from application.documents import DocumentAccessPort
    from application.ports.ml_clients import MLClientPort
    from domain.rag.ports import ChunkRetrievalRepository
    from infrastructure.ml.rag.retrieval_work import RetrievalWork

log = logging.getLogger("default")


async def run_retrieval(query_for_search: str, state: RetrievalWork, ml_clients) -> list[LCDocument]:
    """Search the legal domain first, falling back within the same access scope."""
    filters = [state.retrieval_filter]

    if state.query_domain == DocDomain.LEGAL.value:
        filters.insert(0, with_domain_filter(state.retrieval_filter, DocDomain.LEGAL.value))

    for access_filter in filters:
        candidates = await run_hybrid_search(
            query_for_search,
            state.fetch_k,
            access_filter,
            state.rag,
            ml_clients=ml_clients,
            dense_weight=state.effective_dense_weight,
            sparse_weight=state.effective_sparse_weight,
            visibility_conditions=state.visibility_conditions,
            user_id=state.user.user_id,
            user_group_ids=state.ctx.user_group_ids,
        )

        if candidates:
            return candidates

        if access_filter is not state.retrieval_filter:
            log.info("Legal-filtered retrieval returned 0 candidates — fallback on entire corpus")

    return candidates


async def apply_exact_search(
    query_for_search: str,
    candidates: list[LCDocument],
    user,
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
            limit=5,
            mode=SearchMode.EXACT.value,
            as_of_date=ctx.as_of_date,
        )

        if exact_results:
            existing_hashes = {content_hash(d.page_content) for d in candidates}

            for r in exact_results:
                h = content_hash(r.content)

                if h not in existing_hashes:
                    candidates.append(
                        LCDocument(
                            page_content=r.content,
                            metadata=chunk_result_metadata(r),
                        )
                    )
                    existing_hashes.add(h)

            log.info("Exact-search added %d additional candidates", len(exact_results))
    except Exception as e:
        log.warning("Exact-search failed: %s", e)


async def apply_legal_rerank_fallback(
    query_for_search: str,
    access_filter,
    rag,
    top_k: int,
    docs: list,
    ml_clients,
    *,
    visibility_conditions: list,
    user_id: int,
    user_group_ids: list[int],
    document_access=None,
    user=None,
) -> list:
    """Fallback: rerank entire corpus for legal queries that got no docs."""
    if docs:
        return docs

    log.info("Legal query got no docs after rerank — fallback on entire corpus with rerank")
    fallback_candidates = await run_hybrid_search(
        query_for_search,
        rag.retriever.fetch_k,
        access_filter,
        rag,
        ml_clients=ml_clients,
        visibility_conditions=visibility_conditions,
        user_id=user_id,
        user_group_ids=user_group_ids,
    )

    if document_access is not None:
        fallback_candidates = await filter_documents(fallback_candidates, document_access, user)

    async with ml_clients.reranker_semaphore:
        return await rerank_documents(
            query_for_search,
            fallback_candidates,
            top_n=top_k,
            reranker=ml_clients.reranker(),
            min_score=rag.rerank.min_score,
            score_gap_ratio=rag.rerank.score_gap_ratio,
        )


async def retrieve_with_decomposition(
    query: str, state: RetrievalWork, ml_clients
) -> tuple[list[LCDocument], list[str]]:
    """Run hybrid retrieval with optional multi-query decomposition.

    Returns (candidates, sub_queries).
    """
    if state.rag.features.decomposition_enabled:
        t0 = time.monotonic()
        try:
            sub_queries = await decompose_question(
                query, instructor_client=ml_clients.instructor_client, ml_clients=ml_clients
            )
        except Exception as e:
            log.warning("Decomposition failed, falling back to single query: %s", e)
            sub_queries = [query]

        if len(sub_queries) < 2:
            candidates = await run_retrieval(query, state, ml_clients)
            RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
        else:
            retrieval_tasks = [run_retrieval(sq, state, ml_clients) for sq in sub_queries]
            all_candidates_lists = await asyncio.gather(*retrieval_tasks)
            merged: list[LCDocument] = []

            for c in all_candidates_lists:
                merged.extend(c)

            candidates = deduplicate_docs(merged)

            RAG_DECOMPOSED_TOTAL.inc()
            RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
            log.info(
                "Multi-query decomposition: %d sub-queries -> %d merged candidates (from %d total)",
                len(sub_queries),
                len(candidates),
                sum(len(c) for c in all_candidates_lists),
            )
    else:
        candidates = await run_retrieval(query, state, ml_clients)
        sub_queries = [query]

    for probe in evidence_search_queries(query)[1:]:
        if probe in sub_queries:
            continue

        extra = await run_retrieval(probe, state, ml_clients)
        candidates = deduplicate_docs([*candidates, *extra])
        sub_queries.append(probe)

    return candidates, sub_queries


async def rerank_and_enrich(
    query: str,
    candidates: list[LCDocument],
    state: RetrievalWork,
    ml_clients,
    chunk_search,
    *,
    document_access=None,
) -> tuple[list[tuple[LCDocument, float]], list[LCDocument], float]:
    """Rerank candidates, apply temporal/legal fixes, trim, enrich with neighbors.

    Returns (docs_with_scores, final_docs, avg_sim).
    """
    if state.breadth is None:
        message = "state.breadth must be set before reranking"
        raise ValueError(message)

    candidates = await augment_field_candidates(query, candidates, chunk_search, state.ctx)

    if document_access is not None:
        candidates = await filter_documents(candidates, document_access, state.ctx.to_user_context())

    t0 = time.monotonic()
    async with ml_clients.reranker_semaphore:
        docs = await rerank_documents(
            query,
            candidates,
            top_n=state.rerank_top_n,
            reranker=ml_clients.reranker(),
            min_score=state.rag.rerank.min_score,
            score_gap_ratio=state.rag.rerank.score_gap_ratio,
        )
    RAG_STAGE_DURATION.labels("rerank").observe(time.monotonic() - t0)

    if state.ctx.as_of_date is not None:
        docs = resolve_temporal_conflicts(docs)

    if state.query_domain == DocDomain.LEGAL.value:
        docs = await apply_legal_rerank_fallback(
            query,
            state.retrieval_filter,
            state.rag,
            state.rerank_top_n,
            docs,
            ml_clients,
            visibility_conditions=state.visibility_conditions,
            user_id=state.user.user_id,
            user_group_ids=state.ctx.user_group_ids,
            document_access=document_access,
            user=state.ctx.to_user_context(),
        )

    final_top_k = select_final_top_k(
        state.breadth, enumerate_cases=state.enumerate_cases, rag_settings=state.rag
    )

    if active_evidence.get() is not None:
        capture_exclusions(
            [exclusion_record(item, ExclusionReason.FINAL_TOP_K, "retrieval") for item in docs[final_top_k:]]
        )

    docs = docs[:final_top_k]
    capture_retrieval(docs)
    avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0

    max_context_tokens = compute_context_budget(
        breadth=state.breadth,
        enumerate_cases=state.enumerate_cases,
        history_chars=sum(len(m.content) for m in state.history_messages),
        question_chars=len(state.question),
        num_ctx_narrow=state.rag.llm_num_ctx_narrow,
        num_ctx_broad=state.rag.llm_num_ctx_broad,
        num_predict_narrow=state.rag.llm_num_predict_narrow,
        num_predict_broad=state.rag.llm_num_predict_broad,
    )
    docs = await enrich_with_neighbors(
        docs,
        state.enumerate_cases,
        chunk_search,
        max_context_tokens,
        user=state.ctx.to_user_context(),
        query=query,
    )

    if document_access is not None:
        docs = await filter_scored_documents(docs, document_access, state.ctx.to_user_context())
        avg_sim = sum(score for _, score in docs) / len(docs) if docs else 0.0

    docs = filter_scope_candidates(docs, query, "context")
    docs = await enrich_linked_context(docs, query, chunk_search, state.ctx.to_user_context())
    docs = await enrich_applicable_context(docs, query, chunk_search, state.ctx.to_user_context())
    avg_sim = sum(score for _, score in docs) / len(docs) if docs else 0.0
    docs_only: list[LCDocument] = [d for d, _ in docs] if docs and isinstance(docs[0], tuple) else []
    return docs, docs_only, avg_sim


class RetrievalPipeline:
    def __init__(
        self,
        ml_clients: MLClientPort,
        chunk_search: ChunkRetrievalRepository | None,
        document_access: DocumentAccessPort,
    ) -> None:
        self.ml_clients = ml_clients
        self.chunk_search = chunk_search
        self.document_access = document_access

    async def run(self, query: str, state: RetrievalWork) -> RetrievalWork:
        candidates, state.sub_queries = await retrieve_with_decomposition(query, state, self.ml_clients)

        if state.use_exact_ref_boost:
            await apply_exact_search(query, candidates, state.user, state.ctx, self.chunk_search)

        state.candidates = await filter_documents(candidates, self.document_access, state.user)
        state.enumerate_cases = should_enumerate_cases([doc.page_content for doc in state.candidates])
        state.docs, _, state.avg_sim = await rerank_and_enrich(
            query,
            state.candidates,
            state,
            self.ml_clients,
            self.chunk_search,
            document_access=self.document_access,
        )
        return state
