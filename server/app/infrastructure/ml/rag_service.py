"""Infrastructure implementation of the ChatRAGPort used by ChatService.

Handles the full RAG pipeline: question classification, context retrieval
(Qdrant + BM25 hybrid), reranking, prompt assembly, and LLM streaming.
Exposes ``stream_answer`` as an async iterator of tagged union events
(``TextChunk | SourcesEvent``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from config import settings
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.rag_settings import RagSettings
from domain.value_objects.stream_events import SourcesEvent, StreamEvent, TextChunk
from langchain.schema import Document as LCDocument
from qdrant_client.models import FieldCondition, Filter, MatchValue

from infrastructure.acl import build_qdrant_filter
from infrastructure.clients import (
    get_bm25_index,
    get_embeddings,
    get_llm,
    get_llm_for_breadth,
    get_qdrant_client,
    get_reranker,
    get_vector_store,
)
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
    RAG_STAGE_DURATION,
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

log = logging.getLogger("default")


async def _resolve_hash_to_doc(h: str, access_filter) -> LCDocument | None:
    """Retrieve a document from Qdrant by its content_hash."""
    client = get_qdrant_client()

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


async def _qdrant_dense_search(query: str, k: int, access_filter) -> list[tuple[str, float, LCDocument]]:
    """Search Qdrant directly, returning (content_hash, score, Document) tuples."""
    client = get_qdrant_client()
    embeddings = get_embeddings()

    query_vector = await asyncio.to_thread(embeddings.embed_query, query)
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


class RagService:
    async def stream(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> AsyncIterator[StreamEvent]:
        rag = RagSettings.from_settings()
        t_pipeline_start = time.monotonic()

        user = {"id": ctx.user_id, "kind": ctx.user_kind}
        access_filter = build_qdrant_filter(user, ctx.user_group_ids, ctx.assigned_client_ids)

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
        history_messages = history_to_messages(history_dicts)

        t0 = time.monotonic()
        query_for_search = await condense_question(get_llm(), question, history_messages)
        RAG_STAGE_DURATION.labels("condense").observe(time.monotonic() - t0)

        # --- Semantic answer cache ---
        vis_hash = compute_visibility_scope_hash(ctx.user_kind, ctx.user_group_ids, ctx.assigned_client_ids)
        q_hash = compute_question_hash(query_for_search)
        if rag.cache_enabled:
            t0 = time.monotonic()
            cached = await find_cached_answer(q_hash, vis_hash)
            RAG_STAGE_DURATION.labels("cache_lookup").observe(time.monotonic() - t0)
            if cached is not None:
                RAG_CACHE_HITS_TOTAL.inc()
                log.info("Cache hit for question hash=%s", q_hash[:12])
                answer_text = cached["answer"]
                yield TextChunk(text=answer_text)
                record_rag_answer(breadth="narrow", answer=answer_text, retrieved_count=0, avg_similarity=0.0)
                RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)
                yield SourcesEvent(sources=cached["sources"], confidence=None)
                return
            RAG_CACHE_MISSES_TOTAL.inc()

        # --- Query decomposition ---
        sub_queries = [query_for_search]
        if rag.decomposition_enabled and needs_decomposition(query_for_search):
            t0 = time.monotonic()
            sub_queries = await decompose_question(get_llm(), query_for_search)
            RAG_STAGE_DURATION.labels("decompose").observe(time.monotonic() - t0)
            RAG_DECOMPOSED_TOTAL.inc()

        breadth = (
            ctx.depth if ctx.depth in ("short", "detailed") else classify_question_breadth(query_for_search)
        )
        if breadth == "detailed":
            breadth = "broad"
        elif breadth == "short":
            breadth = "narrow"
        RAG_BREADTH_TOTAL.labels(breadth=breadth).inc()

        fetch_k = rag.retriever_fetch_k_broad if breadth == "broad" else rag.retriever_fetch_k
        top_k = rag.retriever_top_k_broad if breadth == "broad" else rag.retriever_top_k

        prompt = build_prompt(breadth)

        bm25_index = get_bm25_index()

        # --- Retrieval (parallel for decomposed sub-queries) ---
        all_candidates: list = []
        for sq in sub_queries:
            if rag.hybrid_enabled and bm25_index is not None:
                t0 = time.monotonic()
                dense_coro = _qdrant_dense_search(sq, fetch_k, access_filter)
                sparse_coro = asyncio.to_thread(bm25_index.search_with_hashes, sq, fetch_k)
                dense_results, sparse_results = await asyncio.gather(dense_coro, sparse_coro)
                elapsed = time.monotonic() - t0
                RAG_STAGE_DURATION.labels("dense_search").observe(elapsed)
                RAG_STAGE_DURATION.labels("sparse_search").observe(elapsed)
                dense_by_hash = {h: (score, doc) for h, score, doc in dense_results}

                merged_hashes = rrf_merge(
                    [(h, s) for h, s, _ in dense_results],
                    sparse_results,
                    k=rag.rrf_k,
                    dense_weight=rag.dense_weight,
                    sparse_weight=rag.sparse_weight,
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
                        doc = await _resolve_hash_to_doc(h, access_filter)
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
                retriever = get_vector_store().as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": rag.retriever_fetch_k, "filter": access_filter},
                )
                candidates = await asyncio.to_thread(retriever.invoke, sq)
                RAG_STAGE_DURATION.labels("dense_search").observe(time.monotonic() - t0)

            all_candidates.extend(candidates)

        all_candidates = deduplicate_docs(all_candidates)

        t0 = time.monotonic()
        docs = await rerank_documents(
            query_for_search,
            all_candidates,
            top_n=top_k,
            reranker=get_reranker(),
            min_score=rag.rerank_min_score,
            score_gap_ratio=rag.rerank_score_gap_ratio,
        )
        RAG_STAGE_DURATION.labels("rerank").observe(time.monotonic() - t0)

        avg_sim = sum(s for _, s in docs) / len(docs) if docs else 0.0

        # --- Relevance gate ---
        if rag.relevance_gate_enabled:
            t0 = time.monotonic()
            is_relevant, reason = await check_relevance(get_llm(), query_for_search, docs)
            RAG_STAGE_DURATION.labels("relevance_gate").observe(time.monotonic() - t0)

            if not is_relevant:
                RAG_RELEVANCE_GATE_TOTAL.labels(result="rejected").inc()
                log.info("Relevance gate: rejected (%s)", reason)
                not_found_text = "Информация не найдена в документах."
                yield TextChunk(text=not_found_text)
                record_rag_answer(
                    breadth=breadth,
                    answer=not_found_text,
                    retrieved_count=len(docs),
                    avg_similarity=avg_sim,
                )
                RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)
                yield SourcesEvent(sources=[], confidence=0.0)
                return
            RAG_RELEVANCE_GATE_TOTAL.labels(result="passed").inc()

        context = format_docs(docs)
        sources = extract_sources(docs, min_score=rag.source_min_score)

        messages = prompt.format_messages(
            context=context,
            history=history_messages,
            question=question,
        )

        t0 = time.monotonic()
        answer_parts: list[str] = []
        async for chunk in get_llm_for_breadth(breadth).astream(messages):
            text = chunk.content
            if text:
                answer_parts.append(text)
                yield TextChunk(text=text)
        RAG_STAGE_DURATION.labels("generate").observe(time.monotonic() - t0)

        full_answer = "".join(answer_parts)

        if rag.citation_filter_enabled and sources:
            sources = filter_cited_sources(full_answer, sources)

        record_rag_answer(
            breadth=breadth,
            answer=full_answer,
            retrieved_count=len(docs),
            avg_similarity=avg_sim,
        )
        RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)

        confidence = min(1.0, max(0.0, avg_sim))

        # --- Store in cache ---
        if rag.cache_enabled:
            doc_ids = []
            for _, d in docs:
                did = d.metadata.get("document_id")
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

        yield SourcesEvent(sources=sources, confidence=confidence)

    async def invoke(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> tuple[str, list[dict]]:
        answer_parts: list[str] = []
        sources: list[dict] = []

        async for event in self.stream(question, history, ctx):
            if isinstance(event, SourcesEvent):
                sources = event.sources
            elif isinstance(event, TextChunk):
                answer_parts.append(event.text)

        return "".join(answer_parts), sources
