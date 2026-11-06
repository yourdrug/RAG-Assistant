"""Retrieval orchestration: search, reranking, authorized context enrichment."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, NotRequired, TypedDict, Unpack, cast

from langchain.schema import Document as LCDocument

from domain.knowledge.enums import SearchMode
from domain.rag.classification import should_enumerate_cases
from domain.rag.evidence.context import content_with_parent_scope
from domain.rag.retrieval import ContextBudgetOptions, compute_context_budget, select_final_top_k
from domain.shared.hashing import content_hash
from infrastructure.domain_profile.answer_policy import policy_for
from infrastructure.metrics.metrics import RAG_DECOMPOSED_TOTAL, RAG_STAGE_DURATION
from infrastructure.ml.evidence_profiles.profiles import active_evidence_profile
from infrastructure.ml.rag.benchmark_evidence import (
    active_evidence,
    authorize_capture,
    begin_attempt,
    capture_exclusions,
    capture_retrieval,
    capture_stage,
    record_origins,
)
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata
from infrastructure.ml.rag.context_selection import ExclusionReason, exclusion_record
from infrastructure.ml.rag.document_access import filter_documents, filter_scored_documents
from infrastructure.ml.rag.rag_postprocess import enrich_with_neighbors, resolve_temporal_conflicts
from infrastructure.ml.rag.rag_prompts import decompose_question
from infrastructure.ml.rag.rag_reranking import deduplicate_docs, filter_scope_candidates, rerank_documents
from infrastructure.ml.rag.rag_retrieval import run_hybrid_search
from infrastructure.repositories.vector.acl import with_domain_filter


if TYPE_CHECKING:
    from qdrant_client.models import Filter

    from application.documents import DocumentAccessPort
    from application.ports.ml_clients import MLClientPort
    from domain.identity.user_context import UserContext
    from domain.knowledge.access_control import VisibilityCondition
    from domain.rag.context import RagContext
    from domain.rag.ports import ChunkRetrievalRepository
    from domain.rag.settings import RagSettings
    from infrastructure.ml.rag.retrieval_work import RetrievalWork

log = logging.getLogger("default")


async def run_retrieval(
    query_for_search: str, state: RetrievalWork, ml_clients: MLClientPort
) -> list[LCDocument]:
    """Search the legal domain first, falling back within the same access scope."""
    retrieval_filter = cast("Filter | None", state.retrieval_filter)
    filters = [retrieval_filter]

    if policy_for(state.query_domain, getattr(state, "domain_registry", None)).fallback_to_full_corpus:
        filters.insert(0, with_domain_filter(retrieval_filter, state.query_domain))

    candidates: list[LCDocument] = []

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
            user_group_ids=list(state.ctx.user_group_ids),
        )

        record_origins(candidates, "search", query_for_search)

        if candidates:
            return candidates

        if access_filter is not state.retrieval_filter:
            log.info("Legal-filtered retrieval returned 0 candidates — fallback on entire corpus")

    return candidates


async def apply_exact_search(
    query_for_search: str,
    candidates: list[LCDocument],
    user: UserContext,
    ctx: RagContext,
    chunk_search: ChunkRetrievalRepository | None,
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
                    record_origins([candidates[-1]], "exact_search", query_for_search)
                    existing_hashes.add(h)

            log.info("Exact-search added %d additional candidates", len(exact_results))
    except Exception as e:
        log.warning("Exact-search failed: %s", e, exc_info=True)


class LegalFallbackOptions(TypedDict):
    visibility_conditions: list[VisibilityCondition]
    user_id: int
    user_group_ids: list[int]
    document_access: NotRequired[DocumentAccessPort | None]
    user: NotRequired[UserContext | None]


async def apply_legal_rerank_fallback(
    query_for_search: str,
    access_filter: Filter | None,
    rag: RagSettings,
    top_k: int,
    docs: list[tuple[LCDocument, float]],
    *,
    ml_clients: MLClientPort,
    **options: Unpack[LegalFallbackOptions],
) -> list[tuple[LCDocument, float]]:
    """Fallback: rerank entire corpus for legal queries that got no docs."""
    if docs:
        return docs

    visibility_conditions = options["visibility_conditions"]
    user_id = options["user_id"]
    user_group_ids = options["user_group_ids"]
    document_access = options.get("document_access")
    user = options.get("user")

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

    if document_access is not None and user is not None:
        fallback_candidates = await filter_documents(fallback_candidates, document_access, user)

    evidence = active_evidence.get()

    if evidence is not None and evidence.attempts:
        begin_attempt(evidence.attempts[-1]["question"], query_for_search)
        evidence.attempts[-1]["kind"] = "legal_fallback"

    authorize_capture(fallback_candidates)
    record_origins(fallback_candidates, "fallback_search", query_for_search)
    capture_stage("candidates", fallback_candidates, ordered=False)
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
    query: str, state: RetrievalWork, ml_clients: MLClientPort
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
            log.warning("Decomposition failed, falling back to single query: %s", e, exc_info=True)
            sub_queries = [query]

        if len(sub_queries) < MIN_DECOMPOSITION_PARTS:
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

    for probe in active_evidence_profile.get().search_queries(query)[1:]:
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
    ml_clients: MLClientPort,
    chunk_search: ChunkRetrievalRepository | None,
    *,
    document_access: DocumentAccessPort | None = None,
) -> tuple[list[tuple[LCDocument, float]], list[LCDocument], float]:
    """Rerank candidates, apply temporal/legal fixes, trim, enrich with neighbors.

    Returns (docs_with_scores, final_docs, avg_sim).
    """
    if state.breadth is None:
        message = "state.breadth must be set before reranking"
        raise ValueError(message)

    candidates = await active_evidence_profile.get().augment_fields(
        query, candidates, chunk_search, state.ctx
    )

    if document_access is not None:
        candidates = await filter_documents(candidates, document_access, state.ctx.to_user_context())

    authorize_capture(candidates)
    record_origins(candidates, "candidate_augmentation", query)
    capture_stage("candidates", candidates, ordered=False)
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

    docs = retained_temporal_candidates(docs)

    if policy_for(state.query_domain, getattr(state, "domain_registry", None)).fallback_to_full_corpus:
        docs = await apply_legal_rerank_fallback(
            query,
            cast("Filter | None", state.retrieval_filter),
            state.rag,
            state.rerank_top_n,
            docs,
            ml_clients=ml_clients,
            visibility_conditions=state.visibility_conditions,
            user_id=state.user.user_id,
            user_group_ids=list(state.ctx.user_group_ids),
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
    capture_stage("retained", docs, score_kind="reranker")
    capture_retrieval(docs)

    max_context_tokens = compute_context_budget(
        breadth=state.breadth,
        enumerate_cases=state.enumerate_cases,
        history_chars=sum(len(m.content) for m in state.history_messages),
        question_chars=len(state.question),
        num_ctx_narrow=state.rag.llm_num_ctx_narrow,
        num_ctx_broad=state.rag.llm_num_ctx_broad,
        options=ContextBudgetOptions(
            num_predict_narrow=state.rag.llm_num_predict_narrow,
            num_predict_broad=state.rag.llm_num_predict_broad,
        ),
    )
    docs = await enrich_retained_docs(
        docs, state, query, chunk_search, document_access, max_context_tokens=max_context_tokens
    )
    avg_sim = sum(score for _, score in docs) / len(docs) if docs else 0.0
    docs_only: list[LCDocument] = [d for d, _ in docs] if docs and isinstance(docs[0], tuple) else []
    return docs, docs_only, avg_sim


async def enrich_retained_docs(
    docs: list[tuple[LCDocument, float]],
    state: RetrievalWork,
    query: str,
    chunk_search: ChunkRetrievalRepository | None,
    document_access: DocumentAccessPort | None,
    *,
    max_context_tokens: int,
) -> list[tuple[LCDocument, float]]:
    retained_ids = {id(doc) for doc, _ in docs}
    docs = await enrich_with_neighbors(
        docs,
        enumerate_cases=state.enumerate_cases,
        chunk_search=chunk_search,
        max_context_tokens=max_context_tokens,
        user=state.ctx.to_user_context(),
        query=query,
    )

    if document_access is not None:
        docs = await filter_scored_documents(docs, document_access, state.ctx.to_user_context())

    authorize_capture(docs)
    record_origins([item for item in docs if id(item[0]) not in retained_ids], "neighbor")
    docs = filter_scope_candidates(docs, query, "context")
    before_links = {id(doc) for doc, _ in docs}
    docs = await active_evidence_profile.get().enrich_links(
        docs, query, chunk_search, state.ctx.to_user_context()
    )
    record_origins([item for item in docs if id(item[0]) not in before_links], "linked_fragment")
    before_scope = {id(doc) for doc, _ in docs}
    docs = await active_evidence_profile.get().enrich_scope(
        docs, query, chunk_search, state.ctx.to_user_context()
    )
    record_origins([item for item in docs if id(item[0]) not in before_scope], "scope_context")
    # Link/scope enrichers enforce access internally; capture only their authorized output.
    authorize_capture(docs)
    capture_stage("enriched", docs, ordered=False, score_kind="reranker")
    return docs


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
        begin_attempt(state.question, query)
        candidates, sub_queries = await retrieve_with_decomposition(query, state, self.ml_clients)
        state.sub_queries = list(dict.fromkeys([*state.sub_queries, *sub_queries]))

        evidence = active_evidence.get()

        if evidence is not None:
            evidence.attempts[-1]["sub_queries"] = list(state.sub_queries)

        if state.use_exact_ref_boost:
            await apply_exact_search(query, candidates, state.user, state.ctx, self.chunk_search)

        # Scores from different queries are incomparable: re-authorize and rerank
        # the entire pool, including previously enriched evidence, from scratch.
        pooled = await filter_documents(
            [*candidates, *state.candidates, *(doc for doc, _ in state.docs)],
            self.document_access,
            state.user,
        )
        state.candidates = merge_evidence_candidates(pooled)
        state.enumerate_cases = should_enumerate_cases([doc.page_content for doc in state.candidates])
        state.docs, _, state.avg_sim = await rerank_and_enrich(
            state.evidence_query or query,
            state.candidates,
            state,
            self.ml_clients,
            self.chunk_search,
            document_access=self.document_access,
        )
        return state


MIN_DECOMPOSITION_PARTS = 2


def retained_temporal_candidates(docs: list[tuple[LCDocument, float]]) -> list[tuple[LCDocument, float]]:
    before_temporal = docs
    docs = resolve_temporal_conflicts(docs)
    selected_ids = {id(doc) for doc, _ in docs}
    capture_exclusions(
        [
            exclusion_record(item, ExclusionReason.TEMPORAL_CONFLICT, "temporal")
            for item in before_temporal
            if id(item[0]) not in selected_ids
        ]
    )
    return docs


def merge_evidence_candidates(candidates: list[LCDocument]) -> list[LCDocument]:
    """Keep distinct sources, versions and parent conditions when pooling attempts."""
    seen = set()
    merged = []

    for doc in candidates:
        key = (
            doc.metadata.get("document_id"),
            doc.metadata.get("act_version_id"),
            doc.metadata.get("section") or doc.metadata.get("heading"),
            content_hash(content_with_parent_scope(doc)),
        )

        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return merged
