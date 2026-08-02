"""RAG Service — infrastructure implementation of the rag_service protocol used by chat use cases."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from config import settings
from langchain.schema import Document as LCDocument

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
from infrastructure.ml.hybrid import content_hash, rrf_merge
from infrastructure.ml.metrics import (
    RAG_BREADTH_TOTAL,
    RAG_STAGE_DURATION,
    record_rag_answer,
)
from infrastructure.ml.rag import (
    build_prompt,
    classify_question_breadth,
    condense_question,
    deduplicate_docs,
    extract_sources,
    filter_cited_sources,
    format_docs,
    history_to_messages,
    rerank_documents,
)

log = logging.getLogger("default")


async def _resolve_hash_to_doc(h: str, access_filter) -> LCDocument | None:
    """Retrieve a document from Qdrant by its content_hash."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

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
        user_id: int,
        user_kind: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
        depth: str | None = None,
    ) -> AsyncIterator[str]:
        t_pipeline_start = time.monotonic()

        user = {"id": user_id, "kind": user_kind}
        access_filter = build_qdrant_filter(user, user_group_ids, assigned_client_ids)

        history_dicts = []
        for msg in history:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                content = msg.content
                # Skip very short / garbage messages that might pollute condensation
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

        # --- Adaptive breadth: user override or auto-detect ---
        breadth = depth if depth in ("short", "detailed") else classify_question_breadth(query_for_search)
        if breadth == "detailed":
            breadth = "broad"
        elif breadth == "short":
            breadth = "narrow"
        RAG_BREADTH_TOTAL.labels(breadth=breadth).inc()

        fetch_k = settings.retriever_fetch_k_broad if breadth == "broad" else settings.retriever_fetch_k
        top_k = settings.retriever_top_k_broad if breadth == "broad" else settings.retriever_top_k

        prompt = build_prompt(breadth)

        # --- Hybrid retrieval: dense (Qdrant) + sparse (BM25) with RRF ---
        bm25_index = get_bm25_index()

        if settings.hybrid_enabled and bm25_index is not None:
            # Dense search via Qdrant (returns content_hash + score + doc)
            t0 = time.monotonic()
            dense_results = await _qdrant_dense_search(query_for_search, fetch_k, access_filter)
            RAG_STAGE_DURATION.labels("dense_search").observe(time.monotonic() - t0)
            dense_by_hash = {h: (score, doc) for h, score, doc in dense_results}

            # Sparse search via BM25 (returns content_hash + score)
            t0 = time.monotonic()
            sparse_results = await asyncio.to_thread(bm25_index.search_with_hashes, query_for_search, fetch_k)
            RAG_STAGE_DURATION.labels("sparse_search").observe(time.monotonic() - t0)

            # RRF merge
            merged_hashes = rrf_merge(
                [(h, s) for h, s, _ in dense_results],
                sparse_results,
                k=settings.rrf_k,
                dense_weight=settings.dense_weight,
                sparse_weight=settings.sparse_weight,
            )

            # Resolve hashes to LangChain Documents (prefer dense doc if available)
            candidates = []
            seen_hashes = set()
            for h in merged_hashes:
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                if h in dense_by_hash:
                    candidates.append(dense_by_hash[h][1])
                else:
                    # Sparse-only result — retrieve full doc from Qdrant by hash
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
            # Fallback: dense-only retrieval
            t0 = time.monotonic()
            retriever = get_vector_store().as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.retriever_fetch_k, "filter": access_filter},
            )
            candidates = await asyncio.to_thread(retriever.invoke, query_for_search)
            RAG_STAGE_DURATION.labels("dense_search").observe(time.monotonic() - t0)

        # Deduplicate candidates before reranking to improve context diversity
        candidates = deduplicate_docs(candidates)

        t0 = time.monotonic()
        docs = await rerank_documents(
            query_for_search,
            candidates,
            top_n=top_k,
            reranker=get_reranker(),
            min_score=settings.rerank_min_score,
            score_gap_ratio=settings.rerank_score_gap_ratio,
        )
        RAG_STAGE_DURATION.labels("rerank").observe(time.monotonic() - t0)

        context = format_docs(docs)
        sources = extract_sources(docs, min_score=settings.source_min_score)

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
                yield text
        RAG_STAGE_DURATION.labels("generate").observe(time.monotonic() - t0)

        full_answer = "".join(answer_parts)

        if settings.citation_filter_enabled and sources:
            sources = filter_cited_sources(full_answer, sources)

        # Record pipeline-level metrics
        avg_sim = sum(s for _, s, _ in docs) / len(docs) if docs else 0.0
        record_rag_answer(
            breadth=breadth,
            answer=full_answer,
            retrieved_count=len(docs),
            avg_similarity=avg_sim,
        )
        RAG_STAGE_DURATION.labels("total").observe(time.monotonic() - t_pipeline_start)

        yield f"\n__sources__:{json.dumps(sources, ensure_ascii=False)}"

    async def invoke(
        self,
        question: str,
        history: list,
        user_id: int,
        user_kind: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
        depth: str | None = None,
    ) -> tuple[str, list[dict]]:
        answer_parts: list[str] = []
        sources: list[dict] = []

        async for chunk in self.stream(
            question, history, user_id, user_kind, user_group_ids, assigned_client_ids, depth=depth
        ):
            if chunk.startswith("\n__sources__:"):
                sources = json.loads(chunk.replace("\n__sources__:", ""))
            else:
                answer_parts.append(chunk)

        return "".join(answer_parts), sources
