"""Cache query/chunk cross-encoder scores independently of sweep thresholds."""

from __future__ import annotations

from infrastructure.benchmark.sweep_strategies import ShouldCancel, check_cancelled
from infrastructure.ml.rag.rag_reranking import rerank_documents


async def cache_reranker_scores(
    questions: list[dict],
    dense_cache: dict,
    sparse_cache: dict,
    candidates: dict,
    reranker,
    should_cancel: ShouldCancel | None = None,
) -> dict[str, dict[str, float]]:
    """One inference per question, reused by every retrieval configuration.

    Use production input formatting (filename and section) without top-k or
    score filtering. These are applied separately for each configuration.
    """
    cached = {}
    for question in questions:
        query = question["question"]
        if query in cached:
            continue
        await check_cancelled(should_cancel, f"reranker cache for {query}")
        hashes = dict.fromkeys(
            [h for h, _, _ in dense_cache.get(query, [])] + [h for h, _ in sparse_cache.get(query, [])]
        )
        documents = [candidates[h] for h in hashes if h in candidates]
        hash_by_document = {id(candidates[h]): h for h in hashes if h in candidates}
        ranked = await rerank_documents(query, documents, len(documents), reranker=reranker)
        cached[query] = {hash_by_document[id(doc)]: float(score) for doc, score in ranked}
    return cached
