"""Benchmark retrieval — dense, sparse, hybrid search and reranking."""

import logging
from pathlib import Path

from config import get_setting, settings
from domain.value_objects.llm_provider import LLMProvider
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from infrastructure.bm25.hybrid import content_hash, rrf_merge
from infrastructure.ml.clients.factories import (
    create_embeddings,
    create_qdrant_client,
    create_reranker,
    load_bm25_index,
)
from infrastructure.ml.rag import deduplicate_docs

logger = logging.getLogger("default")


def _search_dense(
    question: str, fetch_k: int
) -> tuple[list[tuple[Document, float]], dict[str, tuple[float, Document]]]:
    """Dense search via Qdrant client. Returns (dense_docs, dense_by_hash)."""
    client = create_qdrant_client()
    embeddings = create_embeddings()
    query_vector = embeddings.embed_query_sync(question)
    dense_results = client.search(
        collection_name=settings.collection_name,
        query_vector=query_vector,
        limit=fetch_k,
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


def _search_sparse(question: str, fetch_k: int) -> list[tuple[str, float]]:
    """BM25 sparse search via loaded index."""
    bm25_index = load_bm25_index()
    if bm25_index is not None:
        return bm25_index.search_with_hashes(question, fetch_k)
    return []


def _merge_and_dedup(
    dense_by_hash: dict[str, tuple[float, Document]],
    sparse_results: list[tuple[str, float]],
    fetch_k: int,
) -> list[Document]:
    """RRF merge + deduplication. Returns deduplicated candidate docs."""
    if sparse_results:
        merged_hashes = rrf_merge(
            [(k, v[0]) for k, v in dense_by_hash.items()],
            sparse_results,
            k=int(get_setting("rag.rrf_k")),
            dense_weight=float(get_setting("rag.dense_weight")),
            sparse_weight=float(get_setting("rag.sparse_weight")),
        )
    else:
        merged_hashes = [h for h, _ in [(k, v[0]) for k, v in dense_by_hash.items()]]

    seen = set()
    candidate_docs: list[Document] = []
    for h in merged_hashes:
        if h in seen:
            continue
        seen.add(h)
        if h in dense_by_hash:
            candidate_docs.append(dense_by_hash[h][1])
        if len(candidate_docs) >= fetch_k:
            break

    return deduplicate_docs(candidate_docs)


def _apply_rerank_filters(
    ranked: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """Apply min_score and score_gap_ratio filters to ranked results."""
    min_score = get_setting("rag.rerank_min_score")
    gap_ratio = get_setting("rag.rerank_score_gap_ratio")

    if min_score is not None:
        min_score = float(min_score)
        ranked = [(d, s) for d, s in ranked if s >= min_score]

    if gap_ratio is not None and ranked:
        gap_ratio = float(gap_ratio)
        top_score = ranked[0][1]
        cutoff = top_score * gap_ratio
        ranked = [(d, s) for d, s in ranked if s >= cutoff]

    return ranked


def retrieve_with_scores_hybrid(question: str, top_k: int, fetch_k: int) -> list[tuple[Document, float]]:
    """Retrieve using the production hybrid pipeline: dense + BM25 + RRF + reranker."""
    _, dense_by_hash = _search_dense(question, fetch_k)
    sparse_results = _search_sparse(question, fetch_k)
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
