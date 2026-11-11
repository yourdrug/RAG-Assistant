"""RAG retrieval — Qdrant dense search, hybrid search, hash resolution.

These are module-level functions (no class dependency) extracted from
rag_service.py to reduce its size and isolate the retrieval I/O layer.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import time
from typing import TYPE_CHECKING, NotRequired, TypedDict, Unpack, cast

from langchain.schema import Document as LCDocument
from qdrant_client.models import FieldCondition, Filter, MatchValue

from application.search import HybridRetriever
from config import get_request_settings
from domain.shared.hashing import content_hash
from infrastructure.corpus.snapshot import pin_corpus_call
from infrastructure.metrics.metrics import RAG_STAGE_DURATION
from infrastructure.repositories.vector.embedding_identity import ensure_embedding_identity
from infrastructure.repositories.vector.sparse_vectors import sparse_query
from infrastructure.resilience.retry import retry_on_transient


if TYPE_CHECKING:
    from collections.abc import Iterator

    from qdrant_client import QdrantClient
    from qdrant_client.models import ScoredPoint

    from application.ports.ml_clients import MLClientPort
    from domain.knowledge.access_control import VisibilityCondition
    from domain.rag.settings import RagSettings

log = logging.getLogger("default")

_retriever = HybridRetriever()

# Per-request memo of query vectors: domain fallback and legal rerank fallback
# re-search the same text, which otherwise costs one embedding call each.
query_vectors: ContextVar[dict[tuple[str | None, str], list[float]] | None] = ContextVar(
    "query_vectors", default=None
)


@contextmanager
def query_vector_scope() -> Iterator[None]:
    token = query_vectors.set({})
    try:
        yield
    finally:
        query_vectors.reset(token)


async def embed_search_query(query: str, ml_clients: MLClientPort) -> list[float]:
    embeddings = ml_clients.embeddings()
    key = (getattr(embeddings, "embedding_identity", None), query)
    cache = query_vectors.get()

    if cache is not None and key in cache:
        return cache[key]

    async with ml_clients.embedding_semaphore:
        # Admission can wait through a pool switch; identity is unchanged by endpoint/pool updates.
        vector = await ml_clients.embeddings().embed_query(query)

    if cache is not None:
        cache[key] = vector

    return vector


@pin_corpus_call
@retry_on_transient
async def resolve_hashes_batch(
    hashes: list[str],
    access_filter: Filter | None,
    ml_clients: MLClientPort,
) -> dict[str, LCDocument]:
    """Batch-resolve multiple content_hashes from Qdrant in a single scroll call.

    Wrapped with ``qdrant_search_semaphore`` to cap concurrent Qdrant requests
    and prevent thundering herd during decomposition retries.
    """
    if not hashes:
        return {}

    if access_filter is None:
        message = (
            "resolve_hashes_batch() requires access_filter for ACL enforcement. "
            "Build one via build_qdrant_filter() before calling."
        )
        raise RuntimeError(message)

    client = cast("QdrantClient", ml_clients.qdrant_client())

    should_conditions: list[FieldCondition] = [
        FieldCondition(
            key="metadata.content_hash",
            match=MatchValue(value=h),
        )
        for h in hashes
    ]

    scroll_filter = Filter(must=[access_filter, Filter(should=list(should_conditions))])

    async with ml_clients.qdrant_search_semaphore:
        results = await asyncio.to_thread(
            client.scroll,
            collection_name=get_request_settings().collection_name,
            scroll_filter=scroll_filter,
            limit=len(hashes),
            with_payload=True,
            timeout=get_request_settings().qdrant_timeout,
        )

    points = results[0]
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
    access_filter: Filter | None,
    ml_clients: MLClientPort,
    score_threshold: float | None = None,
) -> list[tuple[str, float, LCDocument]]:
    """Search Qdrant directly, returning (content_hash, score, Document) tuples.

    Wrapped with ``qdrant_search_semaphore`` to cap concurrent Qdrant requests.
    """
    if access_filter is None:
        message = "qdrant_dense_search() requires access_filter for ACL enforcement."
        raise RuntimeError(message)

    client = cast("QdrantClient", ml_clients.qdrant_client())
    embeddings = ml_clients.embeddings()
    await asyncio.to_thread(
        ensure_embedding_identity, client, getattr(embeddings, "embedding_identity", None)
    )

    query_vector = await embed_search_query(query, ml_clients)

    raw_threshold: object = (
        score_threshold
        if score_threshold is not None
        else get_request_settings().rag_retrieval_score_threshold
    )
    threshold = raw_threshold if isinstance(raw_threshold, (int, float)) else 0.0

    async with ml_clients.qdrant_search_semaphore:
        results = await asyncio.to_thread(
            client.search,
            collection_name=get_request_settings().collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=access_filter,
            score_threshold=threshold if threshold > 0 else None,
            timeout=get_request_settings().qdrant_timeout,
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


def search_sparse_points(
    client: QdrantClient, query: str, k: int, access_filter: Filter | None
) -> list[ScoredPoint]:
    """Return sparse hits with payload; apply the complete ACL/temporal scope before top-k."""
    if access_filter is None:
        message = "Sparse search requires an ACL filter"
        raise RuntimeError(message)

    vector = sparse_query(query)

    if not vector.vector.indices:
        return []

    return client.search(
        collection_name=get_request_settings().collection_name,
        query_vector=vector,
        query_filter=access_filter,
        limit=k,
        with_payload=True,
        timeout=get_request_settings().qdrant_timeout,
    )


@pin_corpus_call
@retry_on_transient
async def qdrant_sparse_search(
    query: str, k: int, access_filter: Filter | None, ml_clients: MLClientPort
) -> list[tuple[str, float, LCDocument]]:
    async with ml_clients.qdrant_search_semaphore:
        points = await asyncio.to_thread(
            search_sparse_points, cast("QdrantClient", ml_clients.qdrant_client()), query, k, access_filter
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


class HybridClientOptions(TypedDict):
    ml_clients: MLClientPort
    dense_weight: NotRequired[float | None]
    sparse_weight: NotRequired[float | None]


@pin_corpus_call
async def run_hybrid_search(
    query: str,
    fetch_k: int,
    access_filter: Filter | None,
    rag: RagSettings,
    *,
    visibility_conditions: list[VisibilityCondition] | None,
    user_id: int | None,
    user_group_ids: list[int] | None,
    **options: Unpack[HybridClientOptions],
) -> list[LCDocument]:
    """Run Qdrant dense+sparse retrieval with identical filters and weighted RRF."""
    if access_filter is None:
        message = "run_hybrid_search() requires access_filter for ACL enforcement."
        raise RuntimeError(message)

    if visibility_conditions is None or user_id is None or user_group_ids is None:
        message = "run_hybrid_search() requires visibility_conditions, user_id and user_group_ids."
        raise RuntimeError(message)

    if not visibility_conditions:
        return []

    ml_clients = options["ml_clients"]
    dense_weight = options.get("dense_weight")
    sparse_weight = options.get("sparse_weight")
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
