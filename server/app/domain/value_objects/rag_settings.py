"""RAG Settings — point-in-time snapshot of dynamic config for a single request.

Captures all RAG-related settings at the start of a request to ensure consistency.
If a ConfigParameterChanged event arrives mid-request, the snapshot remains stable.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import settings


@dataclass(frozen=True)
class RagSettings:
    retriever_fetch_k: int
    retriever_top_k: int
    retriever_fetch_k_broad: int
    retriever_top_k_broad: int
    hybrid_enabled: bool
    bm25_fetch_k: int
    rrf_k: int
    dense_weight: float
    sparse_weight: float
    rerank_min_score: float | None
    rerank_score_gap_ratio: float | None
    source_min_score: float
    citation_filter_enabled: bool
    relevance_gate_enabled: bool
    condense_enabled: bool
    decomposition_enabled: bool
    rolling_summary_enabled: bool
    cache_enabled: bool

    @classmethod
    def from_settings(cls) -> RagSettings:
        return cls(
            retriever_fetch_k=settings.retriever_fetch_k,
            retriever_top_k=settings.retriever_top_k,
            retriever_fetch_k_broad=settings.retriever_fetch_k_broad,
            retriever_top_k_broad=settings.retriever_top_k_broad,
            hybrid_enabled=settings.hybrid_enabled,
            bm25_fetch_k=settings.bm25_fetch_k,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
            rerank_min_score=settings.rerank_min_score,
            rerank_score_gap_ratio=settings.rerank_score_gap_ratio,
            source_min_score=settings.source_min_score,
            citation_filter_enabled=settings.citation_filter_enabled,
            relevance_gate_enabled=settings.relevance_gate_enabled,
            condense_enabled=settings.condense_enabled,
            decomposition_enabled=settings.decomposition_enabled,
            rolling_summary_enabled=settings.rolling_summary_enabled,
            cache_enabled=settings.cache_enabled,
        )
