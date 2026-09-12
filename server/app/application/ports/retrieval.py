"""Retrieval port — abstract interface for hybrid search operations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HybridRetrieverPort(Protocol):
    """Core hybrid retrieval: merge dense+sparse, dedup, filter.

    This port encapsulates the retrieval logic that is shared between
    the production RAG pipeline and the benchmark evaluation pipeline.
    """

    def merge_and_dedup(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        dense_by_hash: dict[str, tuple[float, object]],
        fetch_k: int,
        rrf_k: int,
        dense_weight: float,
        sparse_weight: float,
    ) -> list[object]:
        """RRF merge of dense and sparse results, deduplicated."""
        ...

    def apply_rerank_filters(
        self,
        ranked: list[tuple[object, float]],
        min_score: float | None = None,
        score_gap_ratio: float | None = None,
    ) -> list[tuple[object, float]]:
        """Apply min_score and score_gap_ratio filters to ranked results."""
        ...
