"""Infrastructure implementation of the ChatRAGPort used by ChatService.

Handles the full RAG pipeline: question classification, context retrieval
(Qdrant + BM25 hybrid), reranking, prompt assembly, and LLM streaming.
Exposes ``stream_answer`` as an async iterator of tagged union events
(``TextChunk | SourcesEvent``).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from application.dto.chat_dto import RagResult
from config import settings
from domain.services.rag_policy import (
    SUFFICIENCY_ASSESSMENT_SYSTEM,
    classify_query_domain,
    has_exact_reference,
    should_enumerate_cases,
)
from domain.utils import compute_reranker_score
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import BREADTH_ALIASES, Breadth, LLMProvider
from domain.value_objects.not_found_patterns import NOT_FOUND_PATTERNS
from domain.value_objects.rag_settings import (
    FeatureToggles,
    HybridSearchConfig,
    RagSettings,
    RerankConfig,
    RetrieverConfig,
)
from domain.value_objects.search_mode import SearchMode
from domain.value_objects.stream_events import (
    SourcesEvent,
    StatusEvent,
    StreamEvent,
    TextChunk,
    UsageReport,
)
from langchain.schema import Document as LCDocument
from qdrant_client.models import FieldCondition, Filter, MatchValue

if TYPE_CHECKING:
    from infrastructure.ml.client_registry import MLClientRegistry

from infrastructure.acl import build_qdrant_filter, with_domain_filter, with_temporal_filter
from infrastructure.ml.answer_cache import (
    compute_question_hash,
    compute_visibility_scope_hash,
    find_cached_answer,
    store_cached_answer,
)
from infrastructure.bm25.hybrid import content_hash, rrf_merge
from infrastructure.metrics.metrics import (
    RAG_BREADTH_TOTAL,
    RAG_CACHE_HITS_TOTAL,
    RAG_CACHE_MISSES_TOTAL,
    RAG_DECOMPOSED_TOTAL,
    RAG_RELEVANCE_GATE_TOTAL,
    RAG_SELF_RAG_RETRIES,
    RAG_STAGE_DURATION,
    RAG_TEMPORAL_VERSION_CONFLICT_TOTAL,
    extract_usage_from_langchain,
    record_llm_usage,
    record_rag_answer,
)
from infrastructure.ml.rag import (
    CHARS_PER_TOKEN,
    build_prompt,
    check_relevance,
    classify_question_breadth,
    condense_question,
    decompose_question,
    deduplicate_docs,
    extract_sources,
    filter_cited_sources,
    format_docs,
    group_by_section,
    history_to_messages,
    is_out_of_domain,
    rerank_documents,
)
from infrastructure.ml.llm_schemas import SufficiencyAssessment
from shared import request_id_ctx


def _is_not_found_answer(answer: str) -> bool:
    """Return True if the LLM answer indicates information was not found.

    Uses strict matching against the canonical phrase from the system prompt.
    Also detects answers that are too short to be substantive.
    """
    lower = answer.lower().strip()
    # Exact canonical phrase from _CRITICAL_RULES_BLOCK
    if "информация не найдена в документах" in lower:
        return True
    # Very short answers with no source citations are likely "not found"
    if len(lower) < 50 and not re.search(r"\[\d+\]", answer):
        return any(p in lower for p in NOT_FOUND_PATTERNS)
    return False


def _build_rag_settings() -> RagSettings:
    """Build RagSettings from the global config (infrastructure concern)."""
    return RagSettings(
        retriever=RetrieverConfig(
            fetch_k=settings.retriever_fetch_k,
            top_k=settings.retriever_top_k,
            fetch_k_broad=settings.retriever_fetch_k_broad,
            top_k_broad=settings.retriever_top_k_broad,
        ),
        hybrid_search=HybridSearchConfig(
            enabled=settings.hybrid_enabled,
            bm25_fetch_k=settings.bm25_fetch_k,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
        ),
        rerank=RerankConfig(
            min_score=settings.rerank_min_score,
            score_gap_ratio=settings.rerank_score_gap_ratio,
        ),
        features=FeatureToggles(
            citation_filter_enabled=settings.citation_filter_enabled,
            relevance_gate_enabled=settings.relevance_gate_enabled,
            condense_enabled=settings.condense_enabled,
            decomposition_enabled=settings.decomposition_enabled,
            rolling_summary_enabled=settings.rolling_summary_enabled,
            cache_enabled=settings.cache_enabled,
        ),
        source_min_score=settings.source_min_score,
    )


log = logging.getLogger("default")


async def _resolve_hashes_batch(
    hashes: list[str],
    access_filter,
    ml_clients: MLClientRegistry,
) -> dict[str, LCDocument]:
    """Batch-resolve multiple content_hashes from Qdrant in a single scroll call.

    Returns a dict mapping content_hash -> LCDocument for successfully resolved hashes.
    """
    if not hashes:
        return {}

    client = ml_clients.qdrant_client()

    should_conditions: list[FieldCondition] = [
        FieldCondition(
            key="metadata.content_hash",
            match=MatchValue(value=h),
        )
        for h in hashes
    ]

    if access_filter is not None and access_filter.should:
        scroll_filter = Filter(must=[access_filter, Filter(should=should_conditions)])  # type: ignore[arg-type]
    else:
        scroll_filter = Filter(should=should_conditions)  # type: ignore[arg-type]

    results = await asyncio.to_thread(
        client.scroll,
        collection_name=settings.collection_name,
        scroll_filter=scroll_filter,
        limit=len(hashes),
        with_payload=True,
    )

    points = results[0] if isinstance(results, tuple) else results
    resolved = {}
    for point in points:
        payload = point.payload or {}
        page_content = payload.get("page_content", "")
        metadata = payload.get("metadata", {})
        h = metadata.get("content_hash") or payload.get("content_hash")
        if h:
            resolved[h] = LCDocument(page_content=page_content, metadata=metadata)

    return resolved


async def _qdrant_dense_search(
    query: str, k: int, access_filter, ml_clients: MLClientRegistry
) -> list[tuple[str, float, LCDocument]]:
    """Search Qdrant directly, returning (content_hash, score, Document) tuples."""
    client = ml_clients.qdrant_client()
    embeddings = ml_clients.embeddings()

    query_vector = await embeddings.embed_query(query)
    qdrant_filter = None
    if access_filter and access_filter.should:
        qdrant_filter = access_filter

    results = await asyncio.to_thread(
        client.search,
        collection_name=settings.collection_name,
        query_vector=query_vector,
        limit=k,
        query_filter=qdrant_filter,
    )

    docs = []
    for point in results:
        payload = point.payload or {}
        page_content = payload.get("page_content", "")
        metadata = payload.get("metadata", {})
        h = metadata.get("content_hash") or payload.get("content_hash") or content_hash(page_content)
        doc = LCDocument(page_content=page_content, metadata=metadata)
        docs.append((h, point.score, doc))
    return docs


async def _run_hybrid_search(
    query: str,
    fetch_k: int,
    access_filter,
    rag: RagSettings,
    ml_clients: MLClientRegistry,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> list[LCDocument]:
    """Run hybrid dense+BM25 search and return deduplicated candidates."""
    bm25_index = ml_clients.bm25_index()

    if rag.hybrid_search.enabled and bm25_index is not None:
        t0 = time.monotonic()
        dense_coro = _qdrant_dense_search(query, fetch_k, access_filter, ml_clients)
        sparse_coro = asyncio.to_thread(bm25_index.search_with_hashes, query, fetch_k)
        dense_results, sparse_results = await asyncio.gather(dense_coro, sparse_coro)
        elapsed = time.monotonic() - t0
        RAG_STAGE_DURATION.labels("dense_search").observe(elapsed)
        RAG_STAGE_DURATION.labels("sparse_search").observe(elapsed)
        dense_by_hash = {h: (score, doc) for h, score, doc in dense_results}

        effective_dense = dense_weight if dense_weight is not None else rag.hybrid_search.dense_weight
        effective_sparse = sparse_weight if sparse_weight is not None else rag.hybrid_search.sparse_weight

        merged_hashes = rrf_merge(
            [(h, s) for h, s, _ in dense_results],
            sparse_results,
            k=rag.hybrid_search.rrf_k,
            dense_weight=effective_dense,
            sparse_weight=effective_sparse,
        )

        candidates = []
        seen_hashes = set()
        missing_hashes = []
        for h in merged_hashes:
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            if h in dense_by_hash:
                candidates.append(dense_by_hash[h][1])
            else:
                missing_hashes.append(h)

        if missing_hashes:
            resolved = await _resolve_hashes_batch(missing_hashes, access_filter, ml_clients)
            for h in missing_hashes:
                if h in resolved:
                    candidates.append(resolved[h])

        log.info(
            "Hybrid: dense=%d, sparse=%d, merged=%d candidates",
            len(dense_results),
            len(sparse_results),
            len(candidates),
        )
    else:
        t0 = time.monotonic()
        dense_results = await _qdrant_dense_search(query, fetch_k, access_filter, ml_clients)
        candidates = [doc for _, _, doc in dense_results]
        RAG_STAGE_DURATION.labels("dense_search").observe(time.monotonic() - t0)

    return deduplicate_docs(candidates)


class RagService:
    def __init__(self, ml_clients: MLClientRegistry, chunk_search=None, domain_registry=None) -> None:
        self._ml = ml_clients
        self._chunk_search = chunk_search
        self._domain_registry = domain_registry

    @staticmethod
    def _prepare_history_dicts(history: list) -> list[dict]:
        history_dicts = []
        for msg in history:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                content = msg.content
                if len(content.strip()) < 3:
                    continue
                history_dicts.append(
                    {
                        "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                        "content": content,
                    }
                )
            elif isinstance(msg, dict):
                if len(msg.get("content", "").strip()) >= 3:
                    history_dicts.append(msg)
        return history_dicts

    async def _handle_cache_hit(
        self, cached: dict, q_hash: str, t_pipeline_start: float
    ) -> AsyncIterator[StreamEvent]:
        RAG_CACHE_HITS_TOTAL.inc()
        log.info("Cache hit for question hash=%s", q_hash[:12])
        answer_text = cached["answer"]

        # PII guardrail on cached answers too
        if settings.pii_redaction_enabled:
            from infrastructure.ml.guardrails import get_pii_detector

            detector = get_pii_detector()
            pii_found = detector.scan(answer_text)
            if pii_found:
                answer_text, _ = detector.scan_and_redact(answer_text)
                log.warning("PII detected in cached answer: types=%s", pii_found)

        yield TextChunk(text=answer_text)
        record_rag_answer(
            breadth=Breadth.NARROW.value, answer=answer_text, retrieved_count=0, avg_similarity=0.0
        )
        RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)
        yield SourcesEvent(sources=cached["sources"], confidence=None)

    async def _run_retrieval(
        self,
        query_for_search: str,
        fetch_k: int,
        access_filter,
        rag: RagSettings,
        breadth: Breadth,
        query_domain: str,
        effective_dense_weight: float,
        effective_sparse_weight: float,
    ) -> list[LCDocument]:
        if query_domain == DocDomain.LEGAL.value:
            legal_filter = with_domain_filter(access_filter, DocDomain.LEGAL.value)
            candidates = await _run_hybrid_search(
                query_for_search,
                fetch_k,
                legal_filter,
                rag,
                ml_clients=self._ml,
                dense_weight=effective_dense_weight,
                sparse_weight=effective_sparse_weight,
            )
            if not candidates:
                log.info("Legal-filtered retrieval returned 0 candidates — fallback on entire corpus")
                candidates = await _run_hybrid_search(
                    query_for_search,
                    fetch_k,
                    access_filter,
                    rag,
                    ml_clients=self._ml,
                    dense_weight=effective_dense_weight,
                    sparse_weight=effective_sparse_weight,
                )
        else:
            candidates = await _run_hybrid_search(
                query_for_search,
                fetch_k,
                access_filter,
                rag,
                ml_clients=self._ml,
                dense_weight=effective_dense_weight,
                sparse_weight=effective_sparse_weight,
            )
        return candidates

    async def _apply_exact_search(
        self,
        query_for_search: str,
        candidates: list[LCDocument],
        user: dict,
        ctx: ChatContext,
    ) -> None:
        if self._chunk_search is None:
            return
        try:
            exact_results = await self._chunk_search.search_substring(
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

    async def _handle_relevance_gate(
        self,
        query_for_search: str,
        docs: list,
        breadth: Breadth,
        t_pipeline_start: float,
        rag: RagSettings,
    ) -> bool:
        """Check relevance gate. Returns True if relevant, False if rejected."""
        if not rag.features.relevance_gate_enabled:
            return True
        t0 = time.monotonic()
        async with self._ml.llm_semaphore:
            is_relevant, reason = await check_relevance(self._ml.fast_llm(), query_for_search, docs)
        RAG_STAGE_DURATION.labels("relevance_gate").observe(time.monotonic() - t0)
        if not is_relevant:
            RAG_RELEVANCE_GATE_TOTAL.labels(result="rejected").inc()
            log.info("Relevance gate: rejected (%s)", reason)
            return False
        RAG_RELEVANCE_GATE_TOTAL.labels(result="passed").inc()
        return True

    async def _apply_legal_rerank_fallback(
        self,
        query_for_search: str,
        access_filter,
        rag: RagSettings,
        top_k: int,
        docs: list,
    ) -> list:
        if docs:
            return docs
        log.info("Legal query got no docs after rerank — fallback on entire corpus with rerank")
        fallback_candidates = await _run_hybrid_search(
            query_for_search, rag.retriever.fetch_k, access_filter, rag, ml_clients=self._ml
        )
        return await rerank_documents(
            query_for_search,
            fallback_candidates,
            top_n=top_k,
            reranker=self._ml.reranker(),
            min_score=rag.rerank.min_score,
            score_gap_ratio=rag.rerank.score_gap_ratio,
        )

    async def _store_answer_cache(
        self,
        docs: list,
        query_for_search: str,
        q_hash: str,
        vis_hash: str,
        full_answer: str,
        sources: list[dict],
    ) -> None:
        doc_ids = []
        for doc, _score in docs:
            did = doc.metadata.get("document_id")
            if did is not None:
                doc_ids.append(did)
        await store_cached_answer(
            question_text=query_for_search,
            question_hash=q_hash,
            answer=full_answer,
            sources=sources,
            visibility_scope_hash=vis_hash,
            document_ids=doc_ids,
        )

    async def _decompose_and_retrieve(
        self,
        rag: RagSettings,
        query_for_search: str,
        fetch_k: int,
        retrieval_filter,
        breadth: Breadth,
        query_domain: str,
        effective_dense_weight: float,
        effective_sparse_weight: float,
    ) -> tuple[list[LCDocument], list[str]]:
        """Decompose query, retrieve for each sub-query, merge and deduplicate.

        Returns (merged_candidates, sub_queries).
        If decomposition is disabled or fails, returns single-query retrieval.
        """
        if not rag.features.decomposition_enabled:
            candidates = await self._run_retrieval(
                query_for_search, fetch_k, retrieval_filter, rag,
                breadth, query_domain, effective_dense_weight, effective_sparse_weight,
            )
            return candidates, [query_for_search]

        t0 = time.monotonic()
        try:
            sub_queries = await decompose_question(self._ml.fast_llm(), query_for_search)
        except Exception as e:
            log.warning("Decomposition failed, falling back to single query: %s", e)
            sub_queries = [query_for_search]

        if len(sub_queries) < 2:
            candidates = await self._run_retrieval(
                query_for_search, fetch_k, retrieval_filter, rag,
                breadth, query_domain, effective_dense_weight, effective_sparse_weight,
            )
            RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
            return candidates, sub_queries

        retrieval_tasks = [
            self._run_retrieval(
                sq, fetch_k, retrieval_filter, rag,
                breadth, query_domain, effective_dense_weight, effective_sparse_weight,
            )
            for sq in sub_queries
        ]
        all_candidates_lists = await asyncio.gather(*retrieval_tasks)

        merged: list[LCDocument] = []
        for candidates in all_candidates_lists:
            merged.extend(candidates)
        merged = deduplicate_docs(merged)

        RAG_DECOMPOSED_TOTAL.inc()
        RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
        log.info(
            "Multi-query decomposition: %d sub-queries -> %d merged candidates (from %d total)",
            len(sub_queries), len(merged), sum(len(c) for c in all_candidates_lists),
        )
        return merged, sub_queries

    async def _assess_sufficiency(
        self, question: str, docs: list, llm_client=None, model: str = ""
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

        async with self._ml.llm_semaphore:
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

    def _resolve_breadth(self, ctx: ChatContext, query_for_search: str) -> Breadth:
        raw = ctx.depth if ctx.depth in BREADTH_ALIASES else classify_question_breadth(query_for_search)
        breadth = BREADTH_ALIASES.get(raw) or Breadth(raw)
        RAG_BREADTH_TOTAL.labels(breadth=breadth).inc()
        return breadth

    def _compute_effective_weights(
        self, rag: RagSettings, query_for_search: str
    ) -> tuple[float, float, bool]:
        use_exact_ref_boost = has_exact_reference(query_for_search)
        effective_dense_weight = rag.hybrid_search.dense_weight
        effective_sparse_weight = rag.hybrid_search.sparse_weight
        if use_exact_ref_boost:
            effective_sparse_weight = rag.hybrid_search.sparse_weight * settings.exact_ref_sparse_boost
        return effective_dense_weight, effective_sparse_weight, use_exact_ref_boost

    def _resolve_fetch_top_k(self, rag: RagSettings, breadth: Breadth) -> tuple[int, int]:
        fetch_k = rag.retriever.fetch_k_broad if breadth == Breadth.BROAD else rag.retriever.fetch_k
        top_k = rag.retriever.top_k_broad if breadth == Breadth.BROAD else rag.retriever.top_k
        return fetch_k, top_k

    async def _check_cache(
        self,
        rag: RagSettings,
        q_hash: str,
        vis_hash: str,
        t_pipeline_start: float,
    ) -> dict | None:
        if not rag.features.cache_enabled:
            return None
        t0 = time.monotonic()
        cached = await find_cached_answer(q_hash, vis_hash)
        RAG_STAGE_DURATION.labels("cache_lookup").observe(time.monotonic() - t0)
        if cached is None:
            RAG_CACHE_MISSES_TOTAL.inc()
            return None
        return cached

    def _apply_citation_filter(self, rag: RagSettings, full_answer: str, sources: list[dict]) -> list[dict]:
        """Filter sources to only those cited in the LLM answer."""
        if not rag.features.citation_filter_enabled:
            return sources
        return filter_cited_sources(full_answer, sources)

    async def _reject_not_relevant(
        self,
        breadth: Breadth,
        docs: list,
        avg_sim: float,
        t_pipeline_start: float,
    ) -> AsyncIterator[StreamEvent]:
        """Yield a rejection message when relevance gate fails."""
        yield TextChunk(
            text="К сожалению, я не нашёл релевантной информации "
            "в загруженных документах для ответа на ваш вопрос."
        )
        record_rag_answer(
            breadth=breadth.value,
            answer="",
            retrieved_count=len(docs),
            avg_similarity=avg_sim,
        )
        RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)
        yield SourcesEvent(sources=[], confidence=None)

    async def _post_rerank_adjustments(
        self,
        docs: list,
        query_domain: str,
        query_for_search: str,
        fetch_k: int,
        top_k: int,
        access_filter,
        rag: RagSettings,
    ) -> list:
        if query_domain == "legal":
            docs = await self._apply_legal_rerank_fallback(
                query_for_search,
                access_filter,
                rag,
                top_k,
                docs,
            )
        return docs

    async def _enrich_with_neighbors(
        self,
        docs: list[tuple[LCDocument, float]],
        enumerate_cases: bool,
        max_context_tokens: int = 6000,
    ) -> list[tuple[LCDocument, float]]:
        """Add neighboring chunks from the same document for richer context.

        When enumerate_cases is True, pulls adjacent chunks to give the LLM
        full visibility into conditional rules spread across chunk boundaries.
        Neighbor scores are inherited from the anchor (scaled by 0.9) so they
        don't distort the confidence calculation (which runs before this method).
        Stops adding neighbors when estimated tokens exceed max_context_tokens.
        """
        if not self._chunk_search or not enumerate_cases:
            return docs

        # Filter out None hashes to prevent SQL IN (NULL) issues
        existing_hashes = {h for h in (doc.metadata.get("content_hash") for doc, _ in docs) if h is not None}
        new_docs: list[tuple[LCDocument, float]] = []
        # Estimate current token count from existing docs
        current_chars = sum(len(doc.page_content) for doc, _ in docs)
        max_chars = max_context_tokens * CHARS_PER_TOKEN

        for doc, score in docs:
            chunk_index = doc.metadata.get("chunk_index")
            document_id = doc.metadata.get("document_id")

            if not document_id or chunk_index is None:
                continue

            is_table = doc.metadata.get("content_type") == "table"
            try:
                if is_table:
                    neighbors = await self._chunk_search.get_table_batches(
                        document_id, chunk_index, exclude_hashes=existing_hashes
                    )
                else:
                    neighbors = await self._chunk_search.get_neighbors(
                        document_id, chunk_index, window=3, exclude_hashes=existing_hashes
                    )
            except Exception:
                log.warning(
                    "Failed to fetch neighbors for doc_id=%s chunk_index=%s",
                    document_id,
                    chunk_index,
                    exc_info=True,
                )
                continue

            for n in neighbors:
                if n.content_hash and n.content_hash not in existing_hashes:
                    neighbor_chars = len(n.content)
                    if current_chars + neighbor_chars > max_chars:
                        break
                    neighbor_doc = LCDocument(
                        page_content=n.content,
                        metadata={
                            "source": n.filename,
                            "document_id": n.document_id,
                            "content_hash": n.content_hash,
                            "chunk_index": n.chunk_index,
                            "chunk_id": n.chunk_id,
                            "section": n.section,
                            "heading": n.heading,
                            "heading_level": n.heading_level,
                            "content_type": n.content_type,
                            "doc_title": n.doc_title,
                            "doc_type": n.doc_type,
                        },
                    )
                    new_docs.append((neighbor_doc, score * 0.9))
                    existing_hashes.add(n.content_hash)
                    current_chars += neighbor_chars

        return docs + new_docs

    async def stream(  # noqa: C901
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> AsyncIterator[StreamEvent]:
        rag = _build_rag_settings()
        t_pipeline_start = time.monotonic()

        # Set request_id for tracing through the pipeline
        req_id = request_id_ctx.get("")
        if not req_id:
            req_id = uuid.uuid4().hex[:12]
            request_id_ctx.set(req_id)

        user = {"id": ctx.user_id, "kind": ctx.user_kind, "role": ctx.user_role}
        access_filter = build_qdrant_filter(user, ctx.user_group_ids)
        retrieval_filter = with_temporal_filter(access_filter, ctx.as_of_date)

        history_dicts = self._prepare_history_dicts(history)
        history_messages = history_to_messages(history_dicts)

        # ── Step 1: Query condensation ──────────────────────────────────
        t0 = time.monotonic()
        if rag.features.condense_enabled:
            async with self._ml.llm_semaphore:
                query_for_search = await condense_question(self._ml.fast_llm(), question, history_messages)
        else:
            query_for_search = question
        RAG_STAGE_DURATION.labels("condense").observe(time.monotonic() - t0)

        # ── Step 2: Semantic answer cache ───────────────────────────────
        vis_hash = compute_visibility_scope_hash(ctx.user_kind, ctx.user_id, ctx.user_group_ids)
        q_hash = compute_question_hash(query_for_search)
        cached = await self._check_cache(rag, q_hash, vis_hash, t_pipeline_start)
        if cached is not None:
            async for event in self._handle_cache_hit(cached, q_hash, t_pipeline_start):
                yield event
            return

        # ── Step 3: Out-of-domain rejection ─────────────────────────────
        if is_out_of_domain(query_for_search):
            log.info("Out-of-domain question rejected: %s", query_for_search[:100])
            RAG_RELEVANCE_GATE_TOTAL.labels(result="out_of_domain").inc()
            yield TextChunk(text="Информация не найдена в документах.")
            record_rag_answer(
                breadth=Breadth.NARROW.value,
                answer="",
                retrieved_count=0,
                avg_similarity=0.0,
            )
            RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)
            yield SourcesEvent(sources=[], confidence=None)
            return

        # ── Step 4: Query decomposition + breadth resolution ────────────
        breadth = self._resolve_breadth(ctx, query_for_search)
        yield StatusEvent(stage="searching")
        fetch_k, _ = self._resolve_fetch_top_k(rag, breadth)
        # Always rerank with broad top_n so should_enumerate_cases sees enough context
        rerank_top_n = rag.retriever.top_k_broad

        # ── Step 5: Weight computation + domain classification ──────────
        effective_dense_weight, effective_sparse_weight, use_exact_ref_boost = (
            self._compute_effective_weights(rag, query_for_search)
        )
        query_domain = classify_query_domain(query_for_search)

        # ── Step 6: Hybrid retrieval (with optional decomposition) ─────
        candidates, sub_queries = await self._decompose_and_retrieve(
            rag, query_for_search, fetch_k, retrieval_filter,
            breadth, query_domain, effective_dense_weight, effective_sparse_weight,
        )

        # ── Step 7: Exact-search integration ────────────────────────────
        if use_exact_ref_boost:
            await self._apply_exact_search(query_for_search, candidates, user, ctx)

        # ── Step 7b: Detect conditional rules on FULL candidate set ─────
        # Check before reranking so we don't miss conditional chunks that
        # the reranker might score lower than the top-k threshold.
        candidate_texts = [doc.page_content for doc in candidates]
        enumerate_cases = should_enumerate_cases(query_for_search, candidate_texts)

        # ── Step 8: Reranking ───────────────────────────────────────────
        yield StatusEvent(stage="reranking")
        t0 = time.monotonic()
        docs = await rerank_documents(
            query_for_search,
            candidates,
            top_n=rerank_top_n,
            reranker=self._ml.reranker(),
            min_score=rag.rerank.min_score,
            score_gap_ratio=rag.rerank.score_gap_ratio,
        )
        RAG_STAGE_DURATION.labels("rerank").observe(time.monotonic() - t0)

        if ctx.as_of_date is not None:
            docs = self._resolve_temporal_conflicts(docs)

        docs = await self._post_rerank_adjustments(
            docs,
            query_domain,
            query_for_search,
            fetch_k,
            rerank_top_n,
            retrieval_filter,
            rag,
        )

        # ── Step 8b: Conditional enumeration + neighbor enrichment ──────
        # enumerate_cases was computed on full candidate set (Step 7b)
        if not enumerate_cases:
            # Slice to breadth-appropriate top_k
            final_top_k = rag.retriever.top_k if breadth == Breadth.NARROW else rag.retriever.top_k_broad
            docs = docs[:final_top_k]
        else:
            # Keep reranked docs but cap to avoid context pollution from noise
            docs = docs[: rag.retriever.top_k_broad]

        avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0

        # Compute context budget before enrichment so neighbors respect it
        effective_breadth = Breadth.BROAD if enumerate_cases else breadth
        num_ctx = (
            settings.llm_num_ctx_broad if effective_breadth == Breadth.BROAD else settings.llm_num_ctx_narrow
        )
        history_chars = sum(len(m.content) for m in history_messages)
        question_chars = len(question)
        reserved_chars = history_chars + question_chars + 3000  # system + response margin
        reserved_for_system_and_history = max(reserved_chars // CHARS_PER_TOKEN, 1500)
        max_context_tokens = max(num_ctx - reserved_for_system_and_history, 1000)

        # Enrich with neighboring chunks for conditional-question cases
        docs = await self._enrich_with_neighbors(docs, enumerate_cases, max_context_tokens)

        # ── Step 9: Self-RAG relevance gate + retry loop ────────────────
        MAX_SELF_RAG_RETRIES = 1
        for self_rag_attempt in range(MAX_SELF_RAG_RETRIES + 1):
            is_relevant = await self._handle_relevance_gate(
                query_for_search, docs, breadth, t_pipeline_start, rag
            )
            if is_relevant:
                break

            if self_rag_attempt < MAX_SELF_RAG_RETRIES:
                t0 = time.monotonic()
                assessment = await self._assess_sufficiency(query_for_search, docs)
                RAG_STAGE_DURATION.labels("sufficiency_assess").observe(time.monotonic() - t0)

                if assessment.is_sufficient:
                    log.info("Self-RAG: context deemed sufficient despite relevance gate rejection")
                    break

                if assessment.suggested_refinement:
                    log.info(
                        "Self-RAG: retrying with refined query (attempt %d/%d): %r",
                        self_rag_attempt + 1,
                        MAX_SELF_RAG_RETRIES,
                        assessment.suggested_refinement,
                    )
                    RAG_SELF_RAG_RETRIES.inc()
                    query_for_search = assessment.suggested_refinement
                    candidates = await self._run_retrieval(
                        query_for_search,
                        fetch_k,
                        retrieval_filter,
                        rag,
                        breadth,
                        query_domain,
                        effective_dense_weight,
                        effective_sparse_weight,
                    )
                    docs = await rerank_documents(
                        query_for_search,
                        candidates,
                        top_n=rerank_top_n,
                        reranker=self._ml.reranker(),
                        min_score=rag.rerank.min_score,
                        score_gap_ratio=rag.rerank.score_gap_ratio,
                    )
                    docs = await self._post_rerank_adjustments(
                        docs,
                        query_domain,
                        query_for_search,
                        fetch_k,
                        rerank_top_n,
                        retrieval_filter,
                        rag,
                    )
                    avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0
                    continue

            async for event in self._reject_not_relevant(breadth, docs, avg_sim, t_pipeline_start):
                yield event
            return

        # ── Step 10: Prompt building + LLM generation ───────────────────
        domain_addendum = self._domain_prompt_addendum(query_domain, ctx, effective_breadth)
        prompt = build_prompt(
            effective_breadth,
            summary=ctx.summary,
            domain_addendum=domain_addendum,
            enumerate_cases=enumerate_cases,
        )

        # Refine budget with actual system prompt length
        system_text = prompt.messages[0].content if prompt.messages else ""
        reserved_chars = len(system_text) + history_chars + question_chars + 1000
        reserved_for_system_and_history = max(reserved_chars // CHARS_PER_TOKEN, 1500)
        max_context_tokens = max(num_ctx - reserved_for_system_and_history, 1000)
        docs = group_by_section(docs)
        context = format_docs(docs, max_context_tokens=max_context_tokens)
        messages = prompt.format_messages(context=context, history=history_messages, question=question)

        yield StatusEvent(stage="generating")
        t0 = time.monotonic()
        answer_parts: list[str] = []
        last_chunk = None
        async with self._ml.llm_semaphore:
            async for chunk in self._ml.llm_for_breadth(effective_breadth).astream(messages):
                last_chunk = chunk
                text = chunk.content
                if text:
                    answer_parts.append(text)
                    yield TextChunk(text=text)
        RAG_STAGE_DURATION.labels("generate").observe(time.monotonic() - t0)

        # ── Step 11: Token usage extraction ─────────────────────────────
        usage_report = None
        if last_chunk is not None:
            input_tokens, output_tokens = extract_usage_from_langchain(last_chunk)
            if settings.llm_provider == LLMProvider.OLLAMA:
                model_name = settings.llm_model
            else:
                model_name = settings.openrouter_model
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

        full_answer = "".join(answer_parts)

        # ── Step 12: PII guardrail ──────────────────────────────────────
        if settings.pii_redaction_enabled:
            from infrastructure.ml.guardrails import get_pii_detector

            detector = get_pii_detector()
            pii_found = detector.scan(full_answer)
            if pii_found:
                full_answer, _ = detector.scan_and_redact(full_answer)
                log.warning(
                    "PII detected in LLM output [request_id=%s]: types=%s",
                    request_id_ctx.get(""),
                    pii_found,
                )

        # ── Step 13: Source extraction + metrics ────────────────────────
        if _is_not_found_answer(full_answer):
            sources: list[dict] = []
        else:
            sources = extract_sources(docs, min_score=rag.source_min_score)
            sources = self._apply_citation_filter(rag, full_answer, sources)

        record_rag_answer(
            breadth=breadth.value,
            answer=full_answer,
            retrieved_count=len(docs),
            avg_similarity=avg_sim,
        )
        RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)

        confidence = min(1.0, max(0.0, avg_sim))

        # ── Step 14: Cache store + final yield ──────────────────────────
        if rag.features.cache_enabled:
            await self._store_answer_cache(docs, query_for_search, q_hash, vis_hash, full_answer, sources)

        yield SourcesEvent(sources=sources, confidence=confidence, usage=usage_report)

    def _domain_prompt_addendum(self, query_domain: str, ctx: ChatContext, breadth: Breadth) -> str | None:
        """Domain-profile prompt rules (e.g. temporal rules when as_of_date is set)."""
        if self._domain_registry is None:
            return None
        try:
            profile = self._domain_registry.get(query_domain)
        except KeyError:
            return None
        return profile.prompt_addendum(
            breadth.value if hasattr(breadth, "value") else breadth, as_of_date=ctx.as_of_date
        )

    @staticmethod
    def _resolve_temporal_conflicts(docs: list) -> list:
        """Resolve temporal version conflicts by keeping only the latest version per act.

        When multiple versions of the same act pass the temporal filter (overlapping
        effective dates), keeps only the version with the most recent effective_from.
        Non-versioned docs (no act_id) pass through unchanged.
        """
        from datetime import date as _date

        versions_by_act: dict[int, list[tuple]] = {}
        non_versioned: list[tuple] = []

        for doc, score in docs:
            act_id = doc.metadata.get("act_id")
            act_version_id = doc.metadata.get("act_version_id")
            if act_id is None or act_version_id is None:
                non_versioned.append((doc, score))
                continue
            effective_from = doc.metadata.get("effective_from") or _date.min
            versions_by_act.setdefault(act_id, []).append((doc, score, effective_from))

        resolved = list(non_versioned)
        for act_id, versions in versions_by_act.items():
            if len(versions) == 1:
                resolved.append(versions[0][:2])
                continue
            best = max(versions, key=lambda x: x[2])
            resolved.append(best[:2])
            dropped = len(versions) - 1
            RAG_TEMPORAL_VERSION_CONFLICT_TOTAL.inc()
            log.warning(
                "Temporal conflict act_id=%d: kept version with effective_from=%s, dropped %d older versions",
                act_id, best[2], dropped,
            )

        return resolved

    async def invoke(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> RagResult:
        answer_parts: list[str] = []
        sources: list[dict] = []
        breadth = Breadth.NARROW
        domain = DocDomain.GENERAL.value
        retrieval_count = 0
        reranker_score: float | None = None
        confidence: float | None = None
        usage_report = None

        async for event in self.stream(question, history, ctx):
            if isinstance(event, SourcesEvent):
                sources = event.sources
                confidence = event.confidence
                usage_report = event.usage
            elif isinstance(event, TextChunk):
                answer_parts.append(event.text)

        # Extract metadata from the last stream call for logging
        query_for_search = question
        try:
            breadth = Breadth(classify_question_breadth(query_for_search))
        except Exception:
            pass
        domain = classify_query_domain(query_for_search)
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
            model_used=settings.llm_model,
            input_tokens=usage_report.input_tokens if usage_report else None,
            output_tokens=usage_report.output_tokens if usage_report else None,
        )
