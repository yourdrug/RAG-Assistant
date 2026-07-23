"""RAG Service — infrastructure implementation of the rag_service protocol used by chat use cases."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from config import settings
from langchain.schema import Document as LCDocument

from infrastructure.acl import build_qdrant_filter
from infrastructure.clients import get_bm25_index, get_llm, get_reranker, get_vector_store
from infrastructure.ml.hybrid import content_hash, rrf_merge
from infrastructure.ml.rag import (
    build_prompt,
    condense_question,
    extract_sources,
    format_docs,
    history_to_messages,
    rerank_documents,
)

log = logging.getLogger("default")


def _resolve_hash_to_doc(h: str, access_filter) -> LCDocument | None:
    """Retrieve a document from Qdrant by its content_hash."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    # Search by content_hash in metadata
    results = client.scroll(
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


def _qdrant_dense_search(query: str, k: int, access_filter) -> list[tuple[str, float, LCDocument]]:
    """Search Qdrant directly, returning (content_hash, score, Document) tuples."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    embeddings = get_vector_store().embedding

    query_vector = embeddings.embed_query(query)
    # Apply access filter if it has conditions (check should list for non-empty)
    qdrant_filter = None
    if access_filter and access_filter.should:
        qdrant_filter = access_filter

    results = client.search(
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
    ) -> AsyncIterator[str]:
        user = {"id": user_id, "kind": user_kind}
        access_filter = build_qdrant_filter(user, user_group_ids, assigned_client_ids)

        prompt = build_prompt()

        history_dicts = []
        for msg in history:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                history_dicts.append(
                    {
                        "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                        "content": msg.content,
                    }
                )
            elif isinstance(msg, dict):
                history_dicts.append(msg)
        history_messages = history_to_messages(history_dicts)

        query_for_search = await condense_question(get_llm(), question, history_messages)

        # --- Hybrid retrieval: dense (Qdrant) + sparse (BM25) with RRF ---
        bm25_index = get_bm25_index()

        if settings.hybrid_enabled and bm25_index is not None:
            # Dense search via Qdrant (returns content_hash + score + doc)
            dense_results = _qdrant_dense_search(query_for_search, settings.retriever_fetch_k, access_filter)
            dense_by_hash = {h: (score, doc) for h, score, doc in dense_results}

            # Sparse search via BM25 (returns content_hash + score)
            sparse_results = bm25_index.search_with_hashes(query_for_search, k=settings.bm25_fetch_k)

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
                    doc = _resolve_hash_to_doc(h, access_filter)
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
            retriever = get_vector_store().as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.retriever_fetch_k, "filter": access_filter},
            )
            candidates = retriever.invoke(query_for_search)

        docs = rerank_documents(
            query_for_search, candidates, top_n=settings.retriever_top_k, reranker=get_reranker()
        )

        context = format_docs(docs)
        sources = extract_sources(docs)

        messages = prompt.format_messages(
            context=context,
            history=history_messages,
            question=question,
        )

        async for chunk in get_llm().astream(messages):
            text = chunk.content
            if text:
                yield text

        yield f"\n__sources__:{json.dumps(sources, ensure_ascii=False)}"

    async def invoke(
        self,
        question: str,
        history: list,
        user_id: int,
        user_kind: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> tuple[str, list[dict]]:
        answer_parts: list[str] = []
        sources: list[dict] = []

        async for chunk in self.stream(
            question, history, user_id, user_kind, user_group_ids, assigned_client_ids
        ):
            if chunk.startswith("\n__sources__:"):
                sources = json.loads(chunk.replace("\n__sources__:", ""))
            else:
                answer_parts.append(chunk)

        return "".join(answer_parts), sources
