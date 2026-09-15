"""RAG retrieval — Qdrant dense search, hybrid search, hash resolution.

These are module-level functions (no class dependency) extracted from
rag_service.py to reduce its size and isolate the retrieval I/O layer.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from application.services.retrieval import HybridRetriever
from config import settings
from domain.value_objects.rag_settings import RagSettings
from infrastructure.bm25.hybrid import content_hash
from infrastructure.metrics.metrics import RAG_STAGE_DURATION
from infrastructure.ml.rag.rag_reranking import deduplicate_docs
from infrastructure.resilience.retry import retry_on_transient
from langchain.schema import Document as LCDocument
from qdrant_client.models import FieldCondition, Filter, MatchValue

if TYPE_CHECKING:
    from infrastructure.ml.clients.client_registry import MLClientRegistry

log = logging.getLogger("default")

_retriever = HybridRetriever()


@retry_on_transient
async def resolve_hashes_batch(
    hashes: list[str],
    access_filter,
    ml_clients: "MLClientRegistry",
) -> dict[str, LCDocument]:
    """Batch-resolve multiple content_hashes from Qdrant in a single scroll call.

    Wrapped with ``qdrant_search_semaphore`` to cap concurrent Qdrant requests
    and prevent thundering herd during decomposition retries.
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

    if access_filter is not None:
        scroll_filter = Filter(must=[access_filter, Filter(should=should_conditions)])  # type: ignore[arg-type]
    else:
        scroll_filter = Filter(should=should_conditions)  # type: ignore[arg-type]

    async with ml_clients.qdrant_search_semaphore:
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


@retry_on_transient
async def qdrant_dense_search(
    query: str, k: int, access_filter, ml_clients: "MLClientRegistry"
) -> list[tuple[str, float, LCDocument]]:
    """Search Qdrant directly, returning (content_hash, score, Document) tuples.

    Wrapped with ``qdrant_search_semaphore`` to cap concurrent Qdrant requests.
    """
    client = ml_clients.qdrant_client()
    embeddings = ml_clients.embeddings()

    async with ml_clients.embedding_semaphore:
        query_vector = await embeddings.embed_query(query)
    qdrant_filter = access_filter if access_filter is not None else None

    async with ml_clients.qdrant_search_semaphore:
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


async def run_hybrid_search(
    query: str,
    fetch_k: int,
    access_filter,
    rag: RagSettings,
    ml_clients: "MLClientRegistry",
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    visibility_conditions: list | None = None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
) -> list[LCDocument]:
    """Run hybrid dense+BM25 search and return deduplicated candidates.

    When visibility_conditions is provided, BM25 pre-filters candidates
    by ACL before scoring (defense-in-depth: Qdrant resolve still applies).
    """
    bm25_index = await ml_clients._ensure_bm25_loaded()

    if rag.hybrid_search.enabled and bm25_index is not None:
        t0 = time.monotonic()
        dense_coro = qdrant_dense_search(query, fetch_k, access_filter, ml_clients)

        async def _bm25_search():
            async with ml_clients.bm25_search_semaphore:
                return await asyncio.to_thread(
                    bm25_index.search_with_hashes, query, fetch_k,
                    visibility_conditions=visibility_conditions,
                    user_id=user_id,
                    user_group_ids=user_group_ids or [],
                )

        dense_results, sparse_results = await asyncio.gather(dense_coro, _bm25_search())
        elapsed = time.monotonic() - t0
        RAG_STAGE_DURATION.labels("hybrid_search").observe(elapsed)
        dense_by_hash = {h: (score, doc) for h, score, doc in dense_results}

        effective_dense = dense_weight if dense_weight is not None else rag.hybrid_search.dense_weight
        effective_sparse = sparse_weight if sparse_weight is not None else rag.hybrid_search.sparse_weight

        candidates = _retriever.merge_and_dedup(
            dense_results=[(h, s) for h, s, _ in dense_results],
            sparse_results=sparse_results,
            dense_by_hash=dense_by_hash,
            fetch_k=fetch_k,
            rrf_k=rag.hybrid_search.rrf_k,
            dense_weight=effective_dense,
            sparse_weight=effective_sparse,
        )

        # Resolve any hashes that weren't in dense_by_hash (sparse-only results)
        seen_hashes = {content_hash(doc.page_content) for doc in candidates}
        missing_hashes = [h for h, _ in sparse_results if h not in seen_hashes]
        if missing_hashes:
            resolved = await resolve_hashes_batch(missing_hashes, access_filter, ml_clients)
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
        dense_results = await qdrant_dense_search(query, fetch_k, access_filter, ml_clients)
        candidates = [doc for _, _, doc in dense_results]
        RAG_STAGE_DURATION.labels("dense_search").observe(time.monotonic() - t0)

    return deduplicate_docs(candidates)
