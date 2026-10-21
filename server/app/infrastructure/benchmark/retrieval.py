"""Benchmark retrieval — dense, sparse, hybrid search and reranking."""

import logging
from pathlib import Path

from config import get_setting, settings
from domain.runtime.enums import LLMProvider
from infrastructure.ml.clients.factories import ChatModelRole, build_chat_model
from langchain.schema import Document

from infrastructure.ml.rag.rag_retrieval import search_sparse_points
from application.search import HybridRetriever
from domain.rag.retrieval import prioritize_distinct_provisions
from application.ports.ml_clients import MLClientPort
from infrastructure.repositories.vector.embedding_identity import ensure_embedding_identity

logger = logging.getLogger("default")

_retriever = HybridRetriever()


def search_dense(
    question: str,
    fetch_k: int,
    ml_clients: MLClientPort,
    access_filter=None,
) -> tuple[list[tuple[Document, float]], dict[str, tuple[float, Document]]]:
    """Dense search via Qdrant client. Returns (dense_docs, dense_by_hash).

    When *access_filter* is provided, only documents matching the ACL
    filter are returned (defense-in-depth: same filter used in production
    RAG retrieval).
    """
    client = ml_clients.qdrant_client()
    embeddings = ml_clients.embeddings()

    ensure_embedding_identity(client, embeddings.embedding_identity)
    query_vector = embeddings.embed_query_sync(question)
    dense_results = client.search(
        collection_name=settings.collection_name,
        query_vector=query_vector,
        limit=fetch_k,
        query_filter=access_filter,
    )

    dense_docs: list[tuple[Document, float]] = []
    dense_by_hash: dict[str, tuple[float, Document]] = {}
    for point in dense_results:
        payload = point.payload or {}
        page_content = payload.get("page_content", "")
        metadata = payload.get("metadata", {})
        h = str(point.id)
        doc = Document(page_content=page_content, metadata=metadata)
        dense_docs.append((doc, point.score))
        dense_by_hash[h] = (point.score, doc)
    return dense_docs, dense_by_hash


def search_sparse(
    question: str,
    fetch_k: int,
    ml_clients: MLClientPort,
    visibility_conditions=None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
    *,
    access_filter=None,
    documents=None,
) -> list[tuple[str, float]]:
    """Sparse search with the same mandatory Qdrant scope as dense retrieval."""
    if visibility_conditions is None or user_id is None or user_group_ids is None:
        raise RuntimeError("Sparse search requires a complete user context")
    if not visibility_conditions:
        return []
    points = search_sparse_points(ml_clients.qdrant_client(), question, fetch_k, access_filter)
    results = []
    for point in points:
        payload = point.payload or {}
        doc = Document(page_content=payload.get("page_content", ""), metadata=payload.get("metadata", {}))
        key = str(point.id)
        if documents is not None:
            documents[key] = (point.score, doc)
        results.append((key, point.score))
    return results


def merge_and_dedup(
    dense_by_hash: dict[str, tuple[float, Document]],
    sparse_results: list[tuple[str, float]],
    fetch_k: int,
    rrf_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> list[Document]:
    """RRF merge + deduplication. Returns deduplicated candidate docs."""
    dense_results = [(h, v[0]) for h, v in dense_by_hash.items()]
    return _retriever.merge_and_dedup(
        dense_results=dense_results,
        sparse_results=sparse_results,
        dense_by_hash=dense_by_hash,
        fetch_k=fetch_k,
        rrf_k=rrf_k if rrf_k is not None else int(get_setting("rag.rrf_k")),
        dense_weight=dense_weight if dense_weight is not None else float(get_setting("rag.dense_weight")),
        sparse_weight=sparse_weight if sparse_weight is not None else float(get_setting("rag.sparse_weight")),
    )


def apply_rerank_filters(
    ranked: list[tuple[Document, float]],
    min_score: float | None = None,
    score_gap_ratio: float | None = None,
) -> list[tuple[Document, float]]:
    """Apply min_score and score_gap_ratio filters to ranked results."""
    if min_score is None:
        min_score = get_setting("rag.rerank_min_score")
    if score_gap_ratio is None:
        score_gap_ratio = get_setting("rag.rerank_score_gap_ratio")
    return _retriever.apply_rerank_filters(
        ranked,
        min_score=min_score,
        score_gap_ratio=score_gap_ratio,
    )


def retrieve_with_scores_hybrid(
    question: str,
    top_k: int,
    fetch_k: int,
    ml_clients: MLClientPort,
    access_filter=None,
    visibility_conditions=None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
) -> list[tuple[Document, float]]:
    """Retrieve using the production hybrid pipeline: dense + BM25 + RRF + reranker.

    ACL parameters are forwarded to dense and sparse search to ensure
    benchmark retrieval respects document visibility.
    """
    _, dense_by_hash = search_dense(question, fetch_k, ml_clients, access_filter=access_filter)
    dense_ranks = dict(dense_by_hash)
    sparse_results = search_sparse(
        question,
        fetch_k,
        ml_clients,
        access_filter=access_filter,
        documents=dense_by_hash,
        visibility_conditions=visibility_conditions,
        user_id=user_id,
        user_group_ids=user_group_ids,
    )
    candidate_docs = _retriever.merge_and_dedup(
        dense_results=[(key, score) for key, (score, _) in dense_ranks.items()],
        sparse_results=sparse_results,
        dense_by_hash=dense_by_hash,
        fetch_k=fetch_k,
        rrf_k=int(get_setting("rag.rrf_k")),
        dense_weight=float(get_setting("rag.dense_weight")),
        sparse_weight=float(get_setting("rag.sparse_weight")),
    )

    if not candidate_docs:
        return []

    reranker = ml_clients.reranker()
    pairs = []
    for doc in candidate_docs:
        source = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "")
        doc_name = filename or (Path(source).name if source else "")
        content_with_prefix = f"[{doc_name}] {doc.page_content}" if doc_name else doc.page_content
        pairs.append((question, content_with_prefix))

    scores = reranker.predict_sync(pairs)
    ranked = sorted(zip(candidate_docs, scores, strict=False), key=lambda x: x[1], reverse=True)

    return prioritize_distinct_provisions(apply_rerank_filters(ranked))[:top_k]


def build_llm(model: str, base_url: str, provider: str = LLMProvider.OLLAMA):
    """Build LLM instance based on provider."""
    return build_chat_model(provider, ChatModelRole.BENCHMARK, model_name=model, base_url=base_url)
