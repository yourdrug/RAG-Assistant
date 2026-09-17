"""RAG pipeline configuration — builds RagSettings from a settings port."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.value_objects.rag_settings import (
    FeatureToggles,
    HybridSearchConfig,
    RagSettings,
    RerankConfig,
    RetrieverConfig,
)

if TYPE_CHECKING:
    from application.ports.rag_settings import RagSettingsPort


def build_rag_settings(port: RagSettingsPort | None = None) -> RagSettings:
    """Build RagSettings from a settings port.

    If *port* is ``None``, falls back to reading from the global settings
    singleton (backward-compatible default).
    """
    if port is None:
        from infrastructure.ml.config.settings_adapters import LiveRagSettings

        port = LiveRagSettings()

    return RagSettings(
        retriever=RetrieverConfig(
            fetch_k=port.retriever_fetch_k,
            top_k=port.retriever_top_k,
            fetch_k_broad=port.retriever_fetch_k_broad,
            top_k_broad=port.retriever_top_k_broad,
        ),
        hybrid_search=HybridSearchConfig(
            enabled=port.hybrid_enabled,
            bm25_fetch_k=port.bm25_fetch_k,
            rrf_k=port.rrf_k,
            dense_weight=port.dense_weight,
            sparse_weight=port.sparse_weight,
        ),
        rerank=RerankConfig(
            min_score=port.rerank_min_score,
            score_gap_ratio=port.rerank_score_gap_ratio,
        ),
        features=FeatureToggles(
            citation_filter_enabled=port.citation_filter_enabled,
            relevance_gate_enabled=port.relevance_gate_enabled,
            condense_enabled=port.condense_enabled,
            decomposition_enabled=port.decomposition_enabled,
            rolling_summary_enabled=port.rolling_summary_enabled,
            cache_enabled=port.cache_enabled,
        ),
        source_min_score=port.source_min_score,
        exact_ref_sparse_boost=port.exact_ref_sparse_boost,
        llm_num_ctx_narrow=port.llm_num_ctx_narrow,
        llm_num_ctx_broad=port.llm_num_ctx_broad,
        llm_num_predict_narrow=port.llm_num_predict_narrow,
        llm_num_predict_broad=port.llm_num_predict_broad,
        pii_redaction_enabled=port.pii_redaction_enabled,
    )
