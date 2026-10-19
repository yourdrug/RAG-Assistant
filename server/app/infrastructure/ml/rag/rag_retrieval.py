"""RAG retrieval — Qdrant dense search, hybrid search, hash resolution.

These are module-level functions (no class dependency) extracted from
rag_service.py to reduce its size and isolate the retrieval I/O layer.
"""

import asyncio
from infrastructure.corpus.snapshot import pin_corpus_call
import logging
import time

from application.ports.ml_clients import MLClientPort
from application.search import HybridRetriever
from config import settings
from domain.value_objects.rag_settings import RagSettings
from domain.utils import content_hash
from infrastructure.metrics.metrics import RAG_STAGE_DURATION
from infrastructure.resilience.retry import retry_on_transient
from infrastructure.repositories.vector.embedding_identity import ensure_embedding_identity
from langchain.schema import Document as LCDocument
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint
from infrastructure.repositories.vector.sparse_vectors import sparse_query

log = logging.getLogger("default")

_retriever = HybridRetriever()


@pin_corpus_call
@retry_on_transient
async def resolve_hashes_batch(
    hashes: list[str],
    access_filter,
    ml_clients: MLClientPort,
) -> dict[str, LCDocument]:
    """Batch-resolve multiple content_hashes from Qdrant in a single scroll call.

    Wrapped with ``qdrant_search_semaphore`` to cap concurrent Qdrant requests
    and prevent thundering herd during decomposition retries.
    """
    if not hashes:
        return {}

    if access_filter is None:
        raise RuntimeError(
            "resolve_hashes_batch() requires access_filter for ACL enforcement. "
            "Build one via build_qdrant_filter() before calling."
        )

    client = ml_clients.qdrant_client()

    should_conditions: list[FieldCondition] = [
        FieldCondition(
            key="metadata.content_hash",
            match=MatchValue(value=h),
        )
        for h in hashes
    ]

    scroll_filter = Filter(must=[access_filter, Filter(should=should_conditions)])  # type: ignore[arg-type]

    async with ml_clients.qdrant_search_semaphore:
        results = await asyncio.to_thread(
            client.scroll,
            collection_name=settings.collection_name,
            scroll_filter=scroll_filter,
            limit=len(hashes),
            with_payload=True,
            timeout=settings.qdrant_timeout,
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


@pin_corpus_call
@retry_on_transient
async def qdrant_dense_search(
    query: str,
    k: int,
    access_filter,
    ml_clients: MLClientPort,
    score_threshold: float | None = None,
) -> list[tuple[str, float, LCDocument]]:
    """Search Qdrant directly, returning (content_hash, score, Document) tuples.

    Wrapped with ``qdrant_search_semaphore`` to cap concurrent Qdrant requests.
    """
    if access_filter is None:
        raise RuntimeError("qdrant_dense_search() requires access_filter for ACL enforcement.")

    client = ml_clients.qdrant_client()
    embeddings = ml_clients.embeddings()
    await asyncio.to_thread(
        ensure_embedding_identity, client, getattr(embeddings, "embedding_identity", None)
    )

    async with ml_clients.embedding_semaphore:
        # Admission can wait through a pool switch; identity is unchanged by endpoint/pool updates.
        query_vector = await ml_clients.embeddings().embed_query(query)

    raw_threshold = (
        score_threshold
        if score_threshold is not None
        else getattr(settings, "rag_retrieval_score_threshold", 0.0)
    )
    threshold = raw_threshold if isinstance(raw_threshold, (int, float)) else 0.0

    async with ml_clients.qdrant_search_semaphore:
        results = await asyncio.to_thread(
            client.search,
            collection_name=settings.collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=access_filter,
            score_threshold=threshold if threshold > 0 else None,
            timeout=settings.qdrant_timeout,
        )

    docs = []
    for point in results:
        payload = point.payload or {}
        page_content = payload.get("page_content", "")
        metadata = {**payload.get("metadata", {}), "chunk_id": point.id}
        h = metadata.get("content_hash") or payload.get("content_hash") or content_hash(page_content)
        doc = LCDocument(page_content=page_content, metadata=metadata)
        docs.append((h, point.score, doc))
    return docs


def search_sparse_points(client, query: str, k: int, access_filter) -> list[ScoredPoint]:
    """Return sparse hits with payload; apply the complete ACL/temporal scope before top-k."""
    if access_filter is None:
        raise RuntimeError("Sparse search requires an ACL filter")
    vector = sparse_query(query)
    if not vector.vector.indices:
        return []
    return client.search(
        collection_name=settings.collection_name,
        query_vector=vector,
        query_filter=access_filter,
        limit=k,
        with_payload=True,
        timeout=settings.qdrant_timeout,
    )


@pin_corpus_call
@retry_on_transient
async def qdrant_sparse_search(
    query: str, k: int, access_filter, ml_clients: MLClientPort
) -> list[tuple[str, float, LCDocument]]:
    async with ml_clients.qdrant_search_semaphore:
        points = await asyncio.to_thread(
            search_sparse_points, ml_clients.qdrant_client(), query, k, access_filter
        )
    return [
        (
            str(point.id),
            point.score,
            LCDocument(
                page_content=(point.payload or {}).get("page_content", ""),
                metadata={**(point.payload or {}).get("metadata", {}), "chunk_id": point.id},
            ),
        )
        for point in points
    ]


@pin_corpus_call
async def run_hybrid_search(
    query: str,
    fetch_k: int,
    access_filter,
    rag: RagSettings,
    ml_clients: MLClientPort,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    *,
    visibility_conditions: list,
    user_id: int,
    user_group_ids: list[int],
) -> list[LCDocument]:
    """Run Qdrant dense+sparse retrieval with identical filters and weighted RRF."""
    if access_filter is None:
        raise RuntimeError("run_hybrid_search() requires access_filter for ACL enforcement.")
    if visibility_conditions is None or user_id is None or user_group_ids is None:
        raise RuntimeError("run_hybrid_search() requires visibility_conditions, user_id and user_group_ids.")
    if not visibility_conditions:
        return []
    started = time.monotonic()
    if not rag.hybrid_search.enabled:
        dense = await qdrant_dense_search(query, fetch_k, access_filter, ml_clients)
        RAG_STAGE_DURATION.labels("dense_search").observe(time.monotonic() - started)
        return [doc for _, _, doc in dense]
    dense, sparse = await asyncio.gather(
        qdrant_dense_search(query, fetch_k, access_filter, ml_clients),
        qdrant_sparse_search(query, rag.hybrid_search.bm25_fetch_k, access_filter, ml_clients),
    )
    # Merge by point identity; text deduplication happens after rank fusion.
    dense = [(str(doc.metadata["chunk_id"]), score, doc) for _, score, doc in dense]
    documents = {key: (score, doc) for key, score, doc in [*sparse, *dense]}
    candidates = _retriever.merge_and_dedup(
        dense_results=[(key, score) for key, score, _ in dense],
        sparse_results=[(key, score) for key, score, _ in sparse],
        dense_by_hash=documents,
        fetch_k=fetch_k,
        rrf_k=rag.hybrid_search.rrf_k,
        dense_weight=dense_weight if dense_weight is not None else rag.hybrid_search.dense_weight,
        sparse_weight=sparse_weight if sparse_weight is not None else rag.hybrid_search.sparse_weight,
    )
    RAG_STAGE_DURATION.labels("hybrid_search").observe(time.monotonic() - started)
    return candidates
