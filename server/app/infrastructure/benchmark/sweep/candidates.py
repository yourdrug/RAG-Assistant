"""Cache dense and sparse retrieval candidates within the benchmark ACL scope."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain.schema import Document as LCDocument

from config import VectorStoreSection, process_section
from domain.benchmark.evaluation import (
    has_retrieval_labels,
)
from infrastructure.corpus.snapshot import pin_corpus_call
from infrastructure.ml.clients import factories
from infrastructure.ml.rag.rag_retrieval import search_sparse_points


if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter

    from application.benchmark.sweep_dto import SparseCache
    from application.ports.ml_clients import MLClientPort
    from domain.knowledge.access_control import VisibilityCondition
    from infrastructure.corpus.snapshot import BoundQdrantClient


DenseDocumentCache = dict[str, list[tuple[str, float, LCDocument]]]
DocumentCache = dict[str, LCDocument]

logger = logging.getLogger("default")


@pin_corpus_call
async def cache_candidates(
    questions: list[dict[str, Any]],
    max_fetch_k: int,
    *,
    ml_clients: MLClientPort | None = None,
    access_filter: Filter | None = None,
    visibility_conditions: list[VisibilityCondition] | None = None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
) -> tuple[DenseDocumentCache, SparseCache, DocumentCache]:
    """Phase 1: Cache dense + sparse candidates at max fetch_k.

    ACL parameters are forwarded to Qdrant search and BM25 pre-filter
    to ensure sweep retrieval respects document visibility.
    """
    dense_cache: DenseDocumentCache = {}
    sparse_cache: SparseCache = {}
    all_candidates: dict[str, LCDocument] = {}

    if ml_clients is not None:
        client = cast("QdrantClient | BoundQdrantClient", ml_clients.qdrant_client())
        embeddings = ml_clients.embeddings()
    else:
        client = factories.create_qdrant_client()
        embeddings = factories.create_embeddings()

    if access_filter is None or visibility_conditions is None or user_id is None or user_group_ids is None:
        message = "Sweep retrieval requires ACL scope and a complete user context"
        raise RuntimeError(message)

    if not visibility_conditions:
        return dense_cache, sparse_cache, all_candidates

    for q in questions:
        qtext = q["question"]

        dense_results = []
        query_vector = await asyncio.to_thread(embeddings.embed_query_sync, qtext)
        points = await asyncio.to_thread(
            client.search,
            collection_name=process_section(VectorStoreSection).collection_name,
            query_vector=query_vector,
            limit=max_fetch_k,
            query_filter=access_filter,
        )

        for point in points:
            h = str(point.id)
            doc = candidate_document(point.payload)
            dense_results.append((h, point.score, doc))
            all_candidates[h] = doc

        dense_cache[qtext] = dense_results

        sparse_points = await asyncio.to_thread(
            search_sparse_points, cast("QdrantClient", client), qtext, max_fetch_k, access_filter
        )
        sparse_results = []

        for point in sparse_points:
            key = str(point.id)
            sparse_results.append((key, point.score))
            all_candidates[key] = candidate_document(point.payload)

        sparse_cache[qtext] = sparse_results

    logger.info(
        "Cache built: %d dense, %d sparse, %d unique hashes",
        sum(len(v) for v in dense_cache.values()),
        sum(len(v) for v in sparse_cache.values()),
        len(all_candidates),
    )
    return dense_cache, sparse_cache, all_candidates


__all__ = ["has_retrieval_labels"]


def candidate_document(payload: dict[str, Any] | None) -> LCDocument:
    payload = payload or {}
    return LCDocument(page_content=payload.get("page_content", ""), metadata=payload.get("metadata", {}))
