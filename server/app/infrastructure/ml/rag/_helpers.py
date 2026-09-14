"""RAG pipeline helper functions — retrieval, exact search, legal fallback, sufficiency.

Extracted from ``rag_steps.py`` to reduce its size. These are internal helpers
used by the pipeline step functions; they are NOT part of the public API.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.search_mode import SearchMode
from langchain.schema import Document as LCDocument

from infrastructure.bm25.hybrid import content_hash
from infrastructure.ml.clients.llm_schemas import SufficiencyAssessment
from infrastructure.ml.rag.rag_formatting import format_docs
from infrastructure.ml.rag.rag_reranking import deduplicate_docs, rerank_documents
from infrastructure.ml.rag.rag_retrieval import run_hybrid_search
from infrastructure.ml.rag.rag_postprocess import enrich_with_neighbors, resolve_temporal_conflicts
from infrastructure.ml.rag.rag_prompts import decompose_question
from infrastructure.metrics.metrics import RAG_STAGE_DURATION
from infrastructure.repositories.vector.acl import with_domain_filter

if TYPE_CHECKING:
    pass

log = logging.getLogger("default")


def domain_prompt_addendum(query_domain: str, ctx, breadth: Breadth, domain_registry) -> str | None:
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


async def run_retrieval(
    query_for_search: str,
    fetch_k: int,
    access_filter,
    rag,
    ml_clients,
    breadth: Breadth,
    query_domain: str,
    effective_dense_weight: float,
    effective_sparse_weight: float,
    visibility_conditions: list | None = None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
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
            visibility_conditions=visibility_conditions,
            user_id=user_id,
            user_group_ids=user_group_ids,
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
                visibility_conditions=visibility_conditions,
                user_id=user_id,
                user_group_ids=user_group_ids,
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
            visibility_conditions=visibility_conditions,
            user_id=user_id,
            user_group_ids=user_group_ids,
        )
    return candidates


async def apply_exact_search(
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


async def apply_legal_rerank_fallback(
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


async def assess_sufficiency(
    question: str, docs: list, ml_clients, llm_client=None, model: str = ""
) -> SufficiencyAssessment:
    """Self-RAG: assess whether retrieved context is sufficient to answer."""
    from domain.services.rag_policy import SUFFICIENCY_ASSESSMENT_SYSTEM

    if not docs:
        return SufficiencyAssessment(
            is_sufficient=False,
            reasoning="No documents retrieved",
            suggested_refinement=question,
        )

    from infrastructure.ml.clients.instructor_client import create_llm_instructor_client

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


async def retrieve_with_decomposition(
    query: str,
    fetch_k: int,
    retrieval_filter,
    rag,
    ml_clients,
    breadth: Breadth,
    query_domain: str,
    effective_dense_weight: float,
    effective_sparse_weight: float,
    visibility_conditions: list | None = None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
) -> tuple[list[LCDocument], list[str]]:
    """Run hybrid retrieval with optional multi-query decomposition.

    Returns (candidates, sub_queries).
    """
    if rag.features.decomposition_enabled:
        t0 = time.monotonic()
        try:
            sub_queries = await decompose_question(ml_clients.fast_llm(), query)
        except Exception as e:
            log.warning("Decomposition failed, falling back to single query: %s", e)
            sub_queries = [query]

        if len(sub_queries) < 2:
            candidates = await run_retrieval(
                query,
                fetch_k,
                retrieval_filter,
                rag,
                ml_clients,
                breadth,
                query_domain,
                effective_dense_weight,
                effective_sparse_weight,
                visibility_conditions=visibility_conditions,
                user_id=user_id,
                user_group_ids=user_group_ids,
            )
            RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
        else:
            retrieval_tasks = [
                run_retrieval(
                    sq,
                    fetch_k,
                    retrieval_filter,
                    rag,
                    ml_clients,
                    breadth,
                    query_domain,
                    effective_dense_weight,
                    effective_sparse_weight,
                    visibility_conditions=visibility_conditions,
                    user_id=user_id,
                    user_group_ids=user_group_ids,
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
        candidates = await run_retrieval(
            query,
            fetch_k,
            retrieval_filter,
            rag,
            ml_clients,
            breadth,
            query_domain,
            effective_dense_weight,
            effective_sparse_weight,
            visibility_conditions=visibility_conditions,
            user_id=user_id,
            user_group_ids=user_group_ids,
        )
        sub_queries = [query]

    return candidates, sub_queries


async def rerank_and_enrich(
    query: str,
    candidates: list[LCDocument],
    rag,
    ml_clients,
    breadth: Breadth,
    query_domain: str,
    rerank_top_n: int,
    access_filter,
    ctx,
    history_messages: list,
    question_chars: int,
    chunk_search,
    enumerate_cases: bool,
) -> tuple[list[tuple[LCDocument, float]], list[LCDocument], float]:
    """Rerank candidates, apply temporal/legal fixes, trim, enrich with neighbors.

    Returns (docs_with_scores, final_docs, avg_sim).
    """
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
        docs = await apply_legal_rerank_fallback(
            query,
            access_filter,
            rag,
            rerank_top_n,
            docs,
            ml_clients,
        )

    from domain.services.rag_policy import select_final_top_k, compute_context_budget

    final_top_k = select_final_top_k(breadth, enumerate_cases, rag)
    docs = docs[:final_top_k]
    avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0

    max_context_tokens = compute_context_budget(
        breadth=breadth,
        enumerate_cases=enumerate_cases,
        history_chars=sum(len(m.content) for m in history_messages),
        question_chars=question_chars,
        num_ctx_narrow=rag.llm_num_ctx_narrow,
        num_ctx_broad=rag.llm_num_ctx_broad,
    )
    docs = await enrich_with_neighbors(docs, enumerate_cases, chunk_search, max_context_tokens)

    return docs, [d for d, _ in docs] if docs and isinstance(docs[0], tuple) else docs, avg_sim
