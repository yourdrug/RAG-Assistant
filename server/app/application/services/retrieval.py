"""Shared retrieval logic — merge, dedup, and filter for hybrid search.

This module provides the core retrieval primitives shared between the
production RAG pipeline and the benchmark evaluation pipeline. It
eliminates the 3x duplication of dense+sparse merge, dedup, and
rerank-filter logic.
"""

from __future__ import annotations

from typing import Any

from domain.utils import deduplicate_docs, rrf_merge
from domain.services.retrieval_diversity import prioritize_distinct_provisions


class HybridRetriever:
    """Core hybrid retrieval: merge dense+sparse, dedup, filter."""

    def merge_and_dedup(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        dense_by_hash: dict[str, tuple[float, Any]],
        fetch_k: int,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[Any]:
        """RRF merge of dense and sparse results, deduplicated.

        Args:
            dense_results: List of (content_hash, score) from dense search.
            sparse_results: List of (content_hash, score) from BM25 search.
            dense_by_hash: Mapping content_hash -> (score, Document) for dense results.
            fetch_k: Maximum number of candidates to return.
            rrf_k: RRF constant (typically 60).
            dense_weight: Weight for dense scores in RRF.
            sparse_weight: Weight for sparse scores in RRF.

        Returns:
            Deduplicated documents: first one per structural provision in RRF
            order, then remaining chunks in RRF order, within fetch_k.

        """
        if sparse_results:
            merged_hashes = rrf_merge(
                dense_results,
                sparse_results,
                k=rrf_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        else:
            merged_hashes = [h for h, _ in dense_results]

        seen: set[str] = set()
        candidates: list[Any] = []
        for h in merged_hashes:
            if h in seen:
                continue
            seen.add(h)
            if h in dense_by_hash:
                candidates.append(dense_by_hash[h][1])
        # Apply the budget after structural diversity: repeated sentence splits
        # of one provision must not crowd out another retriever's provision.
        return prioritize_distinct_provisions(deduplicate_docs(candidates))[:fetch_k]

    def apply_rerank_filters(
        self,
        ranked: list[tuple[Any, float]],
        min_score: float | None = None,
        score_gap_ratio: float | None = None,
    ) -> list[tuple[Any, float]]:
        """Apply min_score and score_gap_ratio filters to ranked results.

        This is the single source of truth for rerank filtering, replacing
        the duplicated implementations in rag_reranking.rerank_documents
        and benchmark.retrieval._apply_rerank_filters.
        """
        if min_score is not None:
            ranked = [(d, s) for d, s in ranked if s >= min_score]

        if score_gap_ratio is not None and ranked:
            top_score = ranked[0][1]
            cutoff = top_score * score_gap_ratio
            ranked = [(d, s) for d, s in ranked if s >= cutoff]

        return ranked
