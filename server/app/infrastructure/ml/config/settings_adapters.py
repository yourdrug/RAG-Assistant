"""Live adapters for settings ports — reads from global settings singleton at access time.

Consolidates all settings adapter classes into a single module.
"""

from __future__ import annotations

from config import (
    AppSection,
    ChatSection,
    ChunkingSection,
    HybridSearchSection,
    LLMSection,
    MLProviderSection,
    OCRSection,
    RagFeaturesSection,
    RankingSection,
    RetrievalSection,
    StorageSection,
    VectorStoreSection,
    get_setting,
    process_section,
    request_section,
)


class LivePreviewSettings:
    @property
    def preview_max_size_mb(self) -> int:
        return process_section(StorageSection).preview_max_size_mb

    @property
    def preview_cache_ttl_seconds(self) -> int:
        return process_section(StorageSection).preview_cache_ttl_seconds


class LiveRagSettings:
    """RAG settings adapter — reads from global settings at access time (hot-reload)."""

    @property
    def retriever_fetch_k(self) -> int:
        return request_section(RetrievalSection).retriever_fetch_k

    @property
    def retriever_top_k(self) -> int:
        return request_section(RetrievalSection).retriever_top_k

    @property
    def retriever_fetch_k_broad(self) -> int:
        return request_section(RetrievalSection).retriever_fetch_k_broad

    @property
    def retriever_top_k_broad(self) -> int:
        return request_section(RetrievalSection).retriever_top_k_broad

    @property
    def hybrid_enabled(self) -> bool:
        return request_section(HybridSearchSection).hybrid_enabled

    @property
    def bm25_fetch_k(self) -> int:
        return request_section(HybridSearchSection).bm25_fetch_k

    @property
    def rrf_k(self) -> int:
        return request_section(HybridSearchSection).rrf_k

    @property
    def dense_weight(self) -> float:
        return request_section(HybridSearchSection).dense_weight

    @property
    def sparse_weight(self) -> float:
        return request_section(HybridSearchSection).sparse_weight

    @property
    def rerank_min_score(self) -> float | None:
        return request_section(RankingSection).rerank_min_score

    @property
    def rerank_score_gap_ratio(self) -> float | None:
        return request_section(RankingSection).rerank_score_gap_ratio

    @property
    def citation_filter_enabled(self) -> bool:
        return request_section(RankingSection).citation_filter_enabled

    @property
    def relevance_gate_enabled(self) -> bool:
        return request_section(RagFeaturesSection).relevance_gate_enabled

    @property
    def condense_enabled(self) -> bool:
        return request_section(RagFeaturesSection).condense_enabled

    @property
    def decomposition_enabled(self) -> bool:
        return request_section(RagFeaturesSection).decomposition_enabled

    @property
    def rolling_summary_enabled(self) -> bool:
        return request_section(RagFeaturesSection).rolling_summary_enabled

    @property
    def cache_enabled(self) -> bool:
        return bool(get_setting("cache_enabled"))

    @property
    def source_min_score(self) -> float:
        return request_section(RankingSection).source_min_score

    @property
    def exact_ref_sparse_boost(self) -> float:
        return request_section(RankingSection).exact_ref_sparse_boost

    @property
    def llm_num_ctx_narrow(self) -> int:
        return request_section(LLMSection).llm_num_ctx_narrow

    @property
    def llm_num_ctx_broad(self) -> int:
        return request_section(LLMSection).llm_num_ctx_broad

    @property
    def llm_num_predict_narrow(self) -> int:
        return request_section(LLMSection).llm_num_predict_narrow

    @property
    def llm_num_predict_broad(self) -> int:
        return request_section(LLMSection).llm_num_predict_broad

    @property
    def pii_redaction_enabled(self) -> bool:
        return request_section(RagFeaturesSection).pii_redaction_enabled


class LiveChunkSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def chunk_size(self) -> int:
        return process_section(ChunkingSection).chunk_size

    @property
    def chunk_overlap(self) -> int:
        return process_section(ChunkingSection).chunk_overlap


class LiveChatSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def history_window(self) -> int:
        return process_section(ChatSection).history_window

    @property
    def rolling_summary_enabled(self) -> bool:
        return process_section(RagFeaturesSection).rolling_summary_enabled

    @property
    def curator_scope_max_ids(self) -> int:
        return process_section(ChatSection).curator_scope_max_ids


class LiveHealthSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def version(self) -> str:
        return process_section(AppSection).version

    @property
    def uptime_seconds(self) -> float:
        return process_section(AppSection).uptime_seconds

    @property
    def llm_provider(self) -> str:
        return process_section(LLMSection).llm_provider

    @property
    def ml_provider(self) -> str:
        return process_section(MLProviderSection).ml_provider


class LiveConfigAdminSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def llm_provider(self) -> str:
        return process_section(LLMSection).llm_provider

    @property
    def llm_model(self) -> str:
        return process_section(LLMSection).openrouter_model

    @property
    def tei_embed_url(self) -> str:
        return process_section(MLProviderSection).tei_embed_url

    @property
    def tei_rerank_url(self) -> str:
        return process_section(MLProviderSection).tei_rerank_url

    @property
    def ml_provider(self) -> str:
        return process_section(MLProviderSection).ml_provider

    @property
    def deepinfra_embed_model(self) -> str:
        return process_section(MLProviderSection).deepinfra_embed_model

    @property
    def deepinfra_rerank_model(self) -> str:
        return process_section(MLProviderSection).deepinfra_rerank_model

    @property
    def ocr_engine(self) -> str:
        return process_section(OCRSection).ocr_engine

    @property
    def ocr_enabled(self) -> bool:
        return process_section(OCRSection).ocr_enabled

    @property
    def openrouter_model(self) -> str:
        return process_section(LLMSection).openrouter_model

    @property
    def collection_name(self) -> str:
        return process_section(VectorStoreSection).collection_name


class LiveIngestionSettings:
    """Ingestion settings adapter — reads from global settings at access time."""

    @property
    def s3_bucket(self) -> str:
        return process_section(StorageSection).s3_bucket

    @property
    def embed_dim(self) -> int:
        return process_section(MLProviderSection).embed_dim

    @property
    def collection_name(self) -> str:
        return process_section(VectorStoreSection).collection_name

    @property
    def hybrid_enabled(self) -> bool:
        return process_section(HybridSearchSection).hybrid_enabled

    @property
    def tei_embed_url(self) -> str:
        return process_section(MLProviderSection).tei_embed_url

    @property
    def qdrant_url(self) -> str:
        return process_section(VectorStoreSection).qdrant_url
