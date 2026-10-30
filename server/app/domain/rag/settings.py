"""RAG Settings -- point-in-time snapshot of dynamic config for a single request.

Captures all RAG-related settings at the start of a request to ensure consistency.
If a ConfigParameterChanged event arrives mid-request, the snapshot remains stable.

Grouped into logical sub-configs for readability.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_NUM_CTX_NARROW = 8192
DEFAULT_NUM_CTX_BROAD = 16384
DEFAULT_NUM_PREDICT_NARROW = 400
DEFAULT_NUM_PREDICT_BROAD = 2048


@dataclass(frozen=True)
class RetrieverConfig:
    fetch_k: int
    top_k: int
    fetch_k_broad: int
    top_k_broad: int


@dataclass(frozen=True)
class HybridSearchConfig:
    enabled: bool
    bm25_fetch_k: int
    rrf_k: int
    dense_weight: float
    sparse_weight: float


@dataclass(frozen=True)
class RerankConfig:
    min_score: float | None
    score_gap_ratio: float | None


@dataclass(frozen=True)
class FeatureToggles:
    citation_filter_enabled: bool
    relevance_gate_enabled: bool
    condense_enabled: bool
    decomposition_enabled: bool
    rolling_summary_enabled: bool
    cache_enabled: bool


@dataclass(frozen=True)
class RagSettings:
    retriever: RetrieverConfig
    hybrid_search: HybridSearchConfig
    rerank: RerankConfig
    features: FeatureToggles
    source_min_score: float
    exact_ref_sparse_boost: float = 1.0
    llm_num_ctx_narrow: int = DEFAULT_NUM_CTX_NARROW
    llm_num_ctx_broad: int = DEFAULT_NUM_CTX_BROAD
    llm_num_predict_narrow: int = DEFAULT_NUM_PREDICT_NARROW
    llm_num_predict_broad: int = DEFAULT_NUM_PREDICT_BROAD
    pii_redaction_enabled: bool = False
