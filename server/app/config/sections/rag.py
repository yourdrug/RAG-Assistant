"""RAG retrieval, ranking, hybrid search and feature toggles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from config.ownership import Static


class RetrievalSection(BaseModel):
    evidence_profile: Static[Literal["generic", "belarus"]] = "belarus"
    retriever_fetch_k: int = 15
    retriever_top_k: int = 4
    retriever_fetch_k_broad: int = 40
    retriever_top_k_broad: int = 10
    rag_retrieval_score_threshold: Static[float] = 0.0  # minimum cosine similarity for dense search results


class RankingSection(BaseModel):
    rerank_min_score: float | None = 0.15
    rerank_score_gap_ratio: float | None = 0.1
    source_min_score: float = 0.3
    citation_filter_enabled: bool = True
    exact_ref_sparse_boost: float = 2.5


class HybridSearchSection(BaseModel):
    hybrid_enabled: bool = True
    bm25_fetch_k: int = 25
    rrf_k: int = 30
    dense_weight: float = 1.5
    sparse_weight: float = 0.5


class RagFeaturesSection(BaseModel):
    relevance_gate_enabled: bool = True
    condense_enabled: bool = False
    decomposition_enabled: bool = False
    rolling_summary_enabled: bool = True
    cache_enabled: bool = True
    pii_redaction_enabled: bool = True
