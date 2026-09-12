"""Live adapters for settings ports — reads from global settings singleton at access time.

Consolidates all settings adapter classes into a single module.
"""

from __future__ import annotations

from config import get_setting, settings


class LiveRagSettings:
    """RAG settings adapter — reads from global settings at access time (hot-reload)."""

    @property
    def retriever_fetch_k(self) -> int:
        return settings.retriever_fetch_k

    @property
    def retriever_top_k(self) -> int:
        return settings.retriever_top_k

    @property
    def retriever_fetch_k_broad(self) -> int:
        return settings.retriever_fetch_k_broad

    @property
    def retriever_top_k_broad(self) -> int:
        return settings.retriever_top_k_broad

    @property
    def hybrid_enabled(self) -> bool:
        return settings.hybrid_enabled

    @property
    def bm25_fetch_k(self) -> int:
        return settings.bm25_fetch_k

    @property
    def rrf_k(self) -> int:
        return settings.rrf_k

    @property
    def dense_weight(self) -> float:
        return settings.dense_weight

    @property
    def sparse_weight(self) -> float:
        return settings.sparse_weight

    @property
    def rerank_min_score(self) -> float | None:
        return settings.rerank_min_score

    @property
    def rerank_score_gap_ratio(self) -> float | None:
        return settings.rerank_score_gap_ratio

    @property
    def citation_filter_enabled(self) -> bool:
        return settings.citation_filter_enabled

    @property
    def relevance_gate_enabled(self) -> bool:
        return settings.relevance_gate_enabled

    @property
    def condense_enabled(self) -> bool:
        return settings.condense_enabled

    @property
    def decomposition_enabled(self) -> bool:
        return settings.decomposition_enabled

    @property
    def rolling_summary_enabled(self) -> bool:
        return settings.rolling_summary_enabled

    @property
    def cache_enabled(self) -> bool:
        return get_setting("cache_enabled")

    @property
    def source_min_score(self) -> float:
        return settings.source_min_score

    @property
    def exact_ref_sparse_boost(self) -> float:
        return settings.exact_ref_sparse_boost

    @property
    def llm_num_ctx_narrow(self) -> int:
        return settings.llm_num_ctx_narrow

    @property
    def llm_num_ctx_broad(self) -> int:
        return settings.llm_num_ctx_broad

    @property
    def pii_redaction_enabled(self) -> bool:
        return settings.pii_redaction_enabled


class LiveChunkSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def chunk_size(self) -> int:
        return settings.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return settings.chunk_overlap

    @property
    def legal_chunk_size(self) -> int:
        return settings.legal_chunk_size

    @property
    def legal_chunk_overlap(self) -> int:
        return settings.legal_chunk_overlap


class LiveChatSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def history_window(self) -> int:
        return settings.history_window

    @property
    def rolling_summary_enabled(self) -> bool:
        return settings.rolling_summary_enabled


class LiveHealthSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def version(self) -> str:
        return settings.version

    @property
    def uptime_seconds(self) -> float:
        return settings.uptime_seconds

    @property
    def llm_provider(self) -> str:
        return settings.llm_provider


class LiveConfigAdminSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def llm_provider(self) -> str:
        return settings.llm_provider

    @property
    def llm_model(self) -> str:
        return settings.llm_model

    @property
    def tei_embed_url(self) -> str:
        return settings.tei_embed_url

    @property
    def tei_rerank_url(self) -> str:
        return settings.tei_rerank_url

    @property
    def ml_provider(self) -> str:
        return settings.ml_provider

    @property
    def deepinfra_embed_model(self) -> str:
        return settings.deepinfra_embed_model

    @property
    def deepinfra_rerank_model(self) -> str:
        return settings.deepinfra_rerank_model

    @property
    def ocr_engine(self) -> str:
        return settings.ocr_engine

    @property
    def ocr_enabled(self) -> bool:
        return settings.ocr_enabled

    @property
    def openrouter_model(self) -> str:
        return settings.openrouter_model

    @property
    def collection_name(self) -> str:
        return settings.collection_name
