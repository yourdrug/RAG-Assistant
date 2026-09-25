"""Benchmark retrieval — dense, sparse, hybrid search and reranking."""

import logging
from pathlib import Path

from config import get_setting, settings
from domain.value_objects.llm_provider import LLMProvider
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from application.services.retrieval import HybridRetriever
from domain.utils import content_hash
from infrastructure.ml.clients.factories import (
    create_embeddings,
    create_qdrant_client,
    create_reranker,
    load_bm25_index,
)

logger = logging.getLogger("default")

_retriever = HybridRetriever()


def _search_dense(
    question: str,
    fetch_k: int,
    access_filter=None,
) -> tuple[list[tuple[Document, float]], dict[str, tuple[float, Document]]]:
    """Dense search via Qdrant client. Returns (dense_docs, dense_by_hash).

    When *access_filter* is provided, only documents matching the ACL
    filter are returned (defense-in-depth: same filter used in production
    RAG retrieval).
    """
    client = create_qdrant_client()
    embeddings = create_embeddings()
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
        h = metadata.get("content_hash") or content_hash(page_content)
        doc = Document(page_content=page_content, metadata=metadata)
        dense_docs.append((doc, point.score))
        dense_by_hash[h] = (point.score, doc)
    return dense_docs, dense_by_hash


def _search_sparse(
    question: str,
    fetch_k: int,
    visibility_conditions=None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
) -> list[tuple[str, float]]:
    """BM25 sparse search via loaded index.

    When *visibility_conditions* is provided, candidates are pre-filtered
    by ACL before scoring (defense-in-depth: Qdrant resolve still applies).
    """
    bm25_index = load_bm25_index()
    if bm25_index is not None:
        return bm25_index.search_with_hashes(
            question,
            fetch_k,
            visibility_conditions=visibility_conditions,
            user_id=user_id,
            user_group_ids=user_group_ids or [],
        )
    return []


def _merge_and_dedup(
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


def _apply_rerank_filters(
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
    access_filter=None,
    visibility_conditions=None,
    user_id: int | None = None,
    user_group_ids: list[int] | None = None,
) -> list[tuple[Document, float]]:
    """Retrieve using the production hybrid pipeline: dense + BM25 + RRF + reranker.

    ACL parameters are forwarded to dense and sparse search to ensure
    benchmark retrieval respects document visibility.
    """
    _, dense_by_hash = _search_dense(question, fetch_k, access_filter=access_filter)
    sparse_results = _search_sparse(
        question,
        fetch_k,
        visibility_conditions=visibility_conditions,
        user_id=user_id,
        user_group_ids=user_group_ids,
    )
    candidate_docs = _merge_and_dedup(dense_by_hash, sparse_results, fetch_k)

    if not candidate_docs:
        return []

    reranker = create_reranker()
    pairs = []
    for doc in candidate_docs:
        source = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", "")
        doc_name = filename or (Path(source).name if source else "")
        content_with_prefix = f"[{doc_name}] {doc.page_content}" if doc_name else doc.page_content
        pairs.append((question, content_with_prefix))

    scores = reranker.predict_sync(pairs)
    ranked = sorted(zip(candidate_docs, scores, strict=False), key=lambda x: x[1], reverse=True)[:top_k]

    return _apply_rerank_filters(ranked)


def build_llm(model: str, base_url: str, provider: str = LLMProvider.OLLAMA):
    """Build LLM instance based on provider."""
    if provider == LLMProvider.OPENROUTER:
        return ChatOpenAI(
            model_name=model,
            openai_api_key=SecretStr(settings.openrouter_api_key) if settings.openrouter_api_key else None,
            openai_api_base=settings.openrouter_base_url,
            temperature=0.0,
        )
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,
    )
