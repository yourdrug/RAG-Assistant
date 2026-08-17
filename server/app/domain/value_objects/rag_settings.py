"""RAG Settings — point-in-time snapshot of dynamic config for a single request.

Captures all RAG-related settings at the start of a request to ensure consistency.
If a ConfigParameterChanged event arrives mid-request, the snapshot remains stable.
"""

from __future__ import annotations

from dataclasses import dataclass


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
