"""Infrastructure implementation of the ChatRAGPort used by ChatService.

Handles the full RAG pipeline: question classification, context retrieval
(Qdrant + BM25 hybrid), reranking, prompt assembly, and LLM streaming.
Exposes ``stream_answer`` as an async iterator of tagged union events
(``TextChunk | SourcesEvent``).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from application.dto.chat_dto import RagResult
from config import settings
from domain.services.rag_policy import classify_query_domain, has_exact_reference
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import BREADTH_ALIASES, Breadth, LLMProvider
from domain.value_objects.rag_settings import RagSettings
from domain.value_objects.search_mode import SearchMode
from domain.value_objects.stream_events import SourcesEvent, StreamEvent, TextChunk, UsageReport
from langchain.schema import Document as LCDocument
from qdrant_client.models import FieldCondition, Filter, MatchValue

if TYPE_CHECKING:
    from infrastructure.ml.client_registry import MLClientRegistry

from infrastructure.acl import build_qdrant_filter, with_domain_filter
from infrastructure.ml.answer_cache import (
    compute_question_hash,
    compute_visibility_scope_hash,
    find_cached_answer,
    store_cached_answer,
)
from infrastructure.ml.hybrid import content_hash, rrf_merge
from infrastructure.ml.metrics import (
    RAG_BREADTH_TOTAL,
    RAG_CACHE_HITS_TOTAL,
    RAG_CACHE_MISSES_TOTAL,
    RAG_DECOMPOSED_TOTAL,
    RAG_RELEVANCE_GATE_TOTAL,
    RAG_SELF_RAG_RETRIES,
    RAG_STAGE_DURATION,
    extract_usage_from_langchain,
    record_llm_usage,
    record_rag_answer,
)
from infrastructure.ml.rag import (
    build_prompt,
    check_relevance,
    classify_question_breadth,
    condense_question,
    decompose_question,
    deduplicate_docs,
    extract_sources,
    filter_cited_sources,
    format_docs,
    history_to_messages,
    needs_decomposition,
    rerank_documents,
)
from infrastructure.ml.instructor_client import create_instructor_client
from infrastructure.ml.llm_schemas import DecompositionCheck, SufficiencyAssessment

# Context variable for request tracing — set at API entry point
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def _build_rag_settings() -> RagSettings:
    """Build RagSettings from the global config (infrastructure concern)."""
    return RagSettings(
        retriever_fetch_k=settings.retriever_fetch_k,
        retriever_top_k=settings.retriever_top_k,
        retriever_fetch_k_broad=settings.retriever_fetch_k_broad,
        retriever_top_k_broad=settings.retriever_top_k_broad,
        hybrid_enabled=settings.hybrid_enabled,
        bm25_fetch_k=settings.bm25_fetch_k,
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        rerank_min_score=settings.rerank_min_score,
        rerank_score_gap_ratio=settings.rerank_score_gap_ratio,
        source_min_score=settings.source_min_score,
        citation_filter_enabled=settings.citation_filter_enabled,
        relevance_gate_enabled=settings.relevance_gate_enabled,
        condense_enabled=settings.condense_enabled,
        decomposition_enabled=settings.decomposition_enabled,
        rolling_summary_enabled=settings.rolling_summary_enabled,
        cache_enabled=settings.cache_enabled,
    )


log = logging.getLogger("default")


async def _resolve_hash_to_doc(h: str, access_filter, ml_clients: MLClientRegistry) -> LCDocument | None:
    """Retrieve a document from Qdrant by its content_hash."""
    client = ml_clients.qdrant_client()

    results = await asyncio.to_thread(
        client.scroll,
        collection_name=settings.collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.content_hash",
                    match=MatchValue(value=h),
                )
            ]
        ),
        limit=1,
        with_payload=True,
    )

    points = results[0] if isinstance(results, tuple) else results
    if not points:
        return None

    payload = points[0].payload or {}
    page_content = payload.get("page_content", "")
    metadata = payload.get("metadata", {})
    return LCDocument(page_content=page_content, metadata=metadata)


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

    if rag.hybrid_enabled and bm25_index is not None:
        t0 = time.monotonic()
        dense_coro = _qdrant_dense_search(query, fetch_k, access_filter, ml_clients)
        sparse_coro = asyncio.to_thread(bm25_index.search_with_hashes, query, fetch_k)
        dense_results, sparse_results = await asyncio.gather(dense_coro, sparse_coro)
        elapsed = time.monotonic() - t0
        RAG_STAGE_DURATION.labels("dense_search").observe(elapsed)
        RAG_STAGE_DURATION.labels("sparse_search").observe(elapsed)
        dense_by_hash = {h: (score, doc) for h, score, doc in dense_results}

        effective_dense = dense_weight if dense_weight is not None else rag.dense_weight
        effective_sparse = sparse_weight if sparse_weight is not None else rag.sparse_weight

        merged_hashes = rrf_merge(
            [(h, s) for h, s, _ in dense_results],
            sparse_results,
            k=rag.rrf_k,
            dense_weight=effective_dense,
            sparse_weight=effective_sparse,
        )

        candidates = []
        seen_hashes = set()
        for h in merged_hashes:
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            if h in dense_by_hash:
                candidates.append(dense_by_hash[h][1])
            else:
                doc = await _resolve_hash_to_doc(h, access_filter, ml_clients)
                if doc is not None:
                    candidates.append(doc)

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
    def __init__(self, ml_clients: MLClientRegistry, chunk_search=None) -> None:
        self._ml = ml_clients
        self._chunk_search = chunk_search

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
        if not rag.relevance_gate_enabled:
            return True
        t0 = time.monotonic()
        is_relevant, reason = await check_relevance(self._ml.llm(), query_for_search, docs)
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
            query_for_search, rag.retriever_fetch_k, access_filter, rag, ml_clients=self._ml
        )
        return await rerank_documents(
            query_for_search,
            fallback_candidates,
            top_n=top_k,
            reranker=self._ml.reranker(),
            min_score=rag.rerank_min_score,
            score_gap_ratio=rag.rerank_score_gap_ratio,
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

    async def _maybe_decompose(self, rag: RagSettings, query_for_search: str) -> None:
        if rag.decomposition_enabled:
            use_llm = settings.llm_provider == LLMProvider.OPENROUTER or True
            if use_llm:
                t0 = time.monotonic()
                try:
                    should_decompose, sub_queries = await self._llm_assess_decomposition(query_for_search)
                    if should_decompose and len(sub_queries) >= 2:
                        log.info("LLM decomposition: %r -> %s", query_for_search, sub_queries)
                        RAG_DECOMPOSED_TOTAL.inc()
                    else:
                        log.info("LLM decomposition: not needed for %r", query_for_search)
                except Exception as e:
                    log.warning("LLM decomposition failed, falling back to regex: %s", e)
                    if needs_decomposition(query_for_search):
                        await decompose_question(self._ml.llm(), query_for_search)
                        RAG_DECOMPOSED_TOTAL.inc()
                RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
            else:
                if needs_decomposition(query_for_search):
                    t0 = time.monotonic()
                    await decompose_question(self._ml.llm(), query_for_search)
                    RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
                    RAG_DECOMPOSED_TOTAL.inc()

    async def _llm_assess_decomposition(self, question: str) -> tuple[bool, list[str]]:
        """Use LLM to assess whether a query needs decomposition."""
        from config import settings as _settings

        if _settings.llm_provider == LLMProvider.OPENROUTER:
            client = create_instructor_client(
                base_url=_settings.openrouter_base_url,
                api_key=_settings.openrouter_api_key,
            )
            model = _settings.openrouter_model
        else:
            client = create_instructor_client(
                base_url=f"{_settings.ollama_base_url}/v1",
                api_key="ollama",
            )
            model = _settings.llm_model

        system_msg = (
            "Оцени, является ли вопрос составным (содержит 2+ независимых подтемы).\n"
            "Если да — разбей его на 2-4 независимых подвопроса.\n"
            "Если нет — верни needs_decomposition=false."
        )

        import asyncio

        result = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question},
                ],
                response_model=DecompositionCheck,
                max_retries=3,
            )
        )
        return result.needs_decomposition, result.sub_queries

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

        from config import settings as _settings

        if llm_client is None:
            if _settings.llm_provider == LLMProvider.OPENROUTER:
                llm_client = create_instructor_client(
                    base_url=_settings.openrouter_base_url,
                    api_key=_settings.openrouter_api_key,
                )
                model = _settings.openrouter_model
            else:
                llm_client = create_instructor_client(
                    base_url=f"{_settings.ollama_base_url}/v1",
                    api_key="ollama",
                )
                model = _settings.llm_model

        context = format_docs(docs, max_context_tokens=2000)
        system_msg = (
            "Оцени, достаточно ли контекста для ответа на вопрос.\n"
            "Если достаточно — is_sufficient=true.\n"
            "Если нет — is_sufficient=false и предложи уточнённый поисковый запрос для retry."
        )
        user_msg = f"Вопрос: {question}\n\nКонтекст:\n{context}"

        import asyncio

        result = await asyncio.to_thread(
            lambda: llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                response_model=SufficiencyAssessment,
                max_retries=3,
            )
        )
        return result

    def _resolve_breadth(self, ctx: ChatContext, query_for_search: str) -> Breadth:
        breadth = ctx.depth if ctx.depth in BREADTH_ALIASES else classify_question_breadth(query_for_search)
        breadth = BREADTH_ALIASES.get(breadth) or breadth
        RAG_BREADTH_TOTAL.labels(breadth=breadth).inc()
        return breadth

    def _compute_effective_weights(
        self, rag: RagSettings, query_for_search: str
    ) -> tuple[float, float, bool]:
        use_exact_ref_boost = has_exact_reference(query_for_search)
        effective_dense_weight = rag.dense_weight
        effective_sparse_weight = rag.sparse_weight
        if use_exact_ref_boost:
            effective_sparse_weight = rag.sparse_weight * settings.exact_ref_sparse_boost
        return effective_dense_weight, effective_sparse_weight, use_exact_ref_boost

    def _resolve_fetch_top_k(self, rag: RagSettings, breadth: Breadth) -> tuple[int, int]:
        fetch_k = rag.retriever_fetch_k_broad if breadth == Breadth.BROAD else rag.retriever_fetch_k
        top_k = rag.retriever_top_k_broad if breadth == Breadth.BROAD else rag.retriever_top_k
        return fetch_k, top_k

    async def _check_cache(
        self,
        rag: RagSettings,
        q_hash: str,
        vis_hash: str,
        t_pipeline_start: float,
    ) -> dict | None:
        if not rag.cache_enabled:
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
        if not rag.citation_filter_enabled:
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

    async def stream(  # noqa: C901
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> AsyncIterator[StreamEvent]:
        rag = _build_rag_settings()
        t_pipeline_start = time.monotonic()

        # Set request_id for tracing through the pipeline
        req_id = request_id_var.get("")
        if not req_id:
            import uuid

            req_id = uuid.uuid4().hex[:12]
            request_id_var.set(req_id)

        user = {"id": ctx.user_id, "kind": ctx.user_kind}
        access_filter = build_qdrant_filter(user, ctx.user_group_ids)

        history_dicts = self._prepare_history_dicts(history)
        history_messages = history_to_messages(history_dicts)

        t0 = time.monotonic()
        if rag.condense_enabled:
            query_for_search = await condense_question(self._ml.llm(), question, history_messages)
        else:
            query_for_search = question
        RAG_STAGE_DURATION.labels("condense").observe(time.monotonic() - t0)

        # --- Semantic answer cache ---
        vis_hash = compute_visibility_scope_hash(ctx.user_kind, ctx.user_group_ids)
        q_hash = compute_question_hash(query_for_search)
        cached = await self._check_cache(rag, q_hash, vis_hash, t_pipeline_start)
        if cached is not None:
            async for event in self._handle_cache_hit(cached, q_hash, t_pipeline_start):
                yield event
            return

        # --- Query decomposition ---
        await self._maybe_decompose(rag, query_for_search)

        breadth = self._resolve_breadth(ctx, query_for_search)

        fetch_k, top_k = self._resolve_fetch_top_k(rag, breadth)

        # --- Dense/sparse weight boost for exact references ---
        effective_dense_weight, effective_sparse_weight, use_exact_ref_boost = (
            self._compute_effective_weights(rag, query_for_search)
        )

        # --- Query-time domain classification + soft-priority retrieval ---
        query_domain = classify_query_domain(query_for_search)

        candidates = await self._run_retrieval(
            query_for_search,
            fetch_k,
            access_filter,
            rag,
            breadth,
            query_domain,
            effective_dense_weight,
            effective_sparse_weight,
        )

        # --- Exact-search integration (pg_trgm) for queries with structural references ---
        if use_exact_ref_boost:
            await self._apply_exact_search(query_for_search, candidates, user, ctx)

        # --- Reranking ---
        t0 = time.monotonic()
        docs = await rerank_documents(
            query_for_search,
            candidates,
            top_n=top_k,
            reranker=self._ml.reranker(),
            min_score=rag.rerank_min_score,
            score_gap_ratio=rag.rerank_score_gap_ratio,
        )
        RAG_STAGE_DURATION.labels("rerank").observe(time.monotonic() - t0)

        # --- Post-rerank fallback: if legal query got nothing useful after rerank ---
        docs = await self._post_rerank_adjustments(
            docs,
            query_domain,
            query_for_search,
            fetch_k,
            top_k,
            access_filter,
            rag,
        )

        avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0

        # --- Self-RAG: assess sufficiency and retry once if insufficient ---
        MAX_SELF_RAG_RETRIES = 1
        for self_rag_attempt in range(MAX_SELF_RAG_RETRIES + 1):
            is_relevant = await self._handle_relevance_gate(
                query_for_search, docs, breadth, t_pipeline_start, rag
            )
            if is_relevant:
                break

            # On rejection: assess if we should retry with broader query
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
                        access_filter,
                        rag,
                        breadth,
                        query_domain,
                        effective_dense_weight,
                        effective_sparse_weight,
                    )
                    docs = await rerank_documents(
                        query_for_search,
                        candidates,
                        top_n=top_k,
                        reranker=self._ml.reranker(),
                        min_score=rag.rerank_min_score,
                        score_gap_ratio=rag.rerank_score_gap_ratio,
                    )
                    docs = await self._post_rerank_adjustments(
                        docs,
                        query_domain,
                        query_for_search,
                        fetch_k,
                        top_k,
                        access_filter,
                        rag,
                    )
                    avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0
                    continue

            # Final rejection
            async for event in self._reject_not_relevant(breadth, docs, avg_sim, t_pipeline_start):
                yield event
            return

        # --- Prompt adaptation based on actual context composition ---
        has_legal_context = any((doc.metadata.get("doc_domain") == DocDomain.LEGAL.value) for doc, _ in docs)
        prompt = build_prompt(breadth, has_legal_context=has_legal_context)

        # --- Dynamic context budget ---
        num_ctx = settings.llm_num_ctx_broad if breadth == Breadth.BROAD else settings.llm_num_ctx_narrow
        reserved_for_system_and_history = 2000
        max_context_tokens = max(num_ctx - reserved_for_system_and_history, 1000)

        context = format_docs(docs, max_context_tokens=max_context_tokens)
        sources = extract_sources(docs, min_score=rag.source_min_score)

        messages = prompt.format_messages(
            context=context,
            history=history_messages,
            question=question,
        )

        t0 = time.monotonic()
        answer_parts: list[str] = []
        last_chunk = None
        async for chunk in self._ml.llm_for_breadth(breadth).astream(messages):
            text = chunk.content
            if text:
                answer_parts.append(text)
                last_chunk = chunk
                yield TextChunk(text=text)
        RAG_STAGE_DURATION.labels("generate").observe(time.monotonic() - t0)

        # --- Extract token usage from last chunk ---
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

        # --- PII output guardrail ---
        if settings.pii_redaction_enabled:
            from infrastructure.ml.guardrails import get_pii_detector

            detector = get_pii_detector()
            pii_found = detector.scan(full_answer)
            if pii_found:
                full_answer, _ = detector.scan_and_redact(full_answer)
                log.warning(
                    "PII detected in LLM output [request_id=%s]: types=%s",
                    request_id_var.get(""),
                    pii_found,
                )

        sources = self._apply_citation_filter(rag, full_answer, sources)

        record_rag_answer(
            breadth=breadth.value,
            answer=full_answer,
            retrieved_count=len(docs),
            avg_similarity=avg_sim,
        )
        RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)

        confidence = min(1.0, max(0.0, avg_sim))

        # --- Store in cache ---
        if rag.cache_enabled:
            await self._store_answer_cache(
                docs,
                query_for_search,
                q_hash,
                vis_hash,
                full_answer,
                sources,
            )

        yield SourcesEvent(sources=sources, confidence=confidence, usage=usage_report)

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

        async for event in self.stream(question, history, ctx):
            if isinstance(event, SourcesEvent):
                sources = event.sources
            elif isinstance(event, TextChunk):
                answer_parts.append(event.text)

        # Extract metadata from the last stream call for logging
        query_for_search = question
        try:
            breadth = classify_question_breadth(query_for_search)
        except Exception:
            pass
        domain = classify_query_domain(query_for_search)
        retrieval_count = len(sources)
        if sources:
            scores = [s.get("max_score", 0) for s in sources if isinstance(s, dict)]
            reranker_score = max(scores) if scores else None

        return RagResult(
            answer="".join(answer_parts),
            sources=sources,
            breadth=breadth,
            domain=domain,
            retrieval_count=retrieval_count,
            reranker_score=reranker_score,
            model_used=settings.llm_model,
        )
