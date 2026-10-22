"""Cache dense and sparse retrieval candidates within the benchmark ACL scope."""

from __future__ import annotations
from infrastructure.corpus.snapshot import pin_corpus_call
import logging
import asyncio
from langchain.schema import Document as LCDocument
from infrastructure.ml.rag.rag_retrieval import search_sparse_points
from config import settings
from domain.benchmark.evaluation import (
    has_retrieval_labels as has_retrieval_labels,
)
from infrastructure.ml.clients import factories


logger = logging.getLogger("default")


@pin_corpus_call
async def cache_candidates(
    questions: list[dict],
    max_fetch_k: int,
    ml_clients=None,
    access_filter=None,
    visibility_conditions=None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
) -> tuple[dict, dict, dict]:
    """Phase 1: Cache dense + sparse candidates at max fetch_k.

    ACL parameters are forwarded to Qdrant search and BM25 pre-filter
    to ensure sweep retrieval respects document visibility.
    """
    dense_cache: dict[str, list] = {}
    sparse_cache: dict[str, list] = {}
    all_candidates: dict[str, "LCDocument"] = {}

    if ml_clients is not None:
        client = ml_clients.qdrant_client()
        embeddings = ml_clients.embeddings()
    else:
        client = factories.create_qdrant_client()
        embeddings = factories.create_embeddings()

    if access_filter is None or visibility_conditions is None or user_id is None or user_group_ids is None:
        raise RuntimeError("Sweep retrieval requires ACL scope and a complete user context")

    if not visibility_conditions:
        return dense_cache, sparse_cache, all_candidates

    for q in questions:
        qtext = q["question"]

        dense_results = []
        query_vector = await asyncio.to_thread(embeddings.embed_query_sync, qtext)
        points = await asyncio.to_thread(
            client.search,
            collection_name=settings.collection_name,
            query_vector=query_vector,
            limit=max_fetch_k,
            query_filter=access_filter,
        )

        for point in points:
            payload = point.payload or {}
            page_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            h = str(point.id)

            doc = LCDocument(page_content=page_content, metadata=metadata)
            dense_results.append((h, point.score, doc))
            all_candidates[h] = doc

        dense_cache[qtext] = dense_results

        sparse_points = await asyncio.to_thread(
            search_sparse_points, client, qtext, max_fetch_k, access_filter
        )
        sparse_results = []

        for point in sparse_points:
            payload = point.payload or {}
            key = str(point.id)
            sparse_results.append((key, point.score))
            all_candidates[key] = LCDocument(
                page_content=payload.get("page_content", ""), metadata=payload.get("metadata", {})
            )

        sparse_cache[qtext] = sparse_results

    logger.info(
        "Cache built: %d dense, %d sparse, %d unique hashes",
        sum(len(v) for v in dense_cache.values()),
        sum(len(v) for v in sparse_cache.values()),
        len(all_candidates),
    )
    return dense_cache, sparse_cache, all_candidates
