"""RAG pipeline configuration — builds RagSettings from global config."""

from __future__ import annotations

from config import get_setting, settings
from domain.value_objects.rag_settings import (
    FeatureToggles,
    HybridSearchConfig,
    RagSettings,
    RerankConfig,
    RetrieverConfig,
)


def build_rag_settings() -> RagSettings:
    """Build RagSettings from the global config."""
    return RagSettings(
        retriever=RetrieverConfig(
            fetch_k=settings.retriever_fetch_k,
            top_k=settings.retriever_top_k,
            fetch_k_broad=settings.retriever_fetch_k_broad,
            top_k_broad=settings.retriever_top_k_broad,
        ),
        hybrid_search=HybridSearchConfig(
            enabled=settings.hybrid_enabled,
            bm25_fetch_k=settings.bm25_fetch_k,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.sparse_weight,
        ),
        rerank=RerankConfig(
            min_score=settings.rerank_min_score,
            score_gap_ratio=settings.rerank_score_gap_ratio,
        ),
        features=FeatureToggles(
            citation_filter_enabled=settings.citation_filter_enabled,
            relevance_gate_enabled=settings.relevance_gate_enabled,
            condense_enabled=settings.condense_enabled,
            decomposition_enabled=settings.decomposition_enabled,
            rolling_summary_enabled=settings.rolling_summary_enabled,
            cache_enabled=get_setting("cache_enabled"),
        ),
        source_min_score=settings.source_min_score,
    )
