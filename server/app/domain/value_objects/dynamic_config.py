"""Declarative dynamic config parameters — single source of truth.

Every hot-reloadable config parameter is declared once here. The
``_build_defaults()`` seed function and the ``_DYNAMIC_FIELDS`` mapping
are both derived from this list, eliminating the risk of triple-sync bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import LLMProvider


# Target ownership only: these sets do not activate seeding, updates or hot reload.
# DYNAMIC_PARAMS below remains the legacy registry until each migration stage is ready.
SECRET_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "db_password",
        "qdrant_api_key",
        "openrouter_api_key",
        "deepinfra_api_key",
        "redis_password",
        "s3_access_key",
        "s3_secret_key",
        "jwt_secret_key",
        "admin_password",
    }
)

STATIC_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "ml_internal_allowed_origins",
        "db_host",
        "db_port",
        "db_user",
        "db_name",
        "db_slave_hosts",
        "db_slave_ports",
        "db_pool_size",
        "db_max_overflow",
        "db_pool_recycle_sec",
        "db_statement_timeout_ms",
        "db_idle_in_transaction_timeout_ms",
        "qdrant_timeout",
        "qdrant_search_max_concurrent",
        "qdrant_write_max_concurrent",
        "rag_retrieval_score_threshold",
        "llm_max_concurrent",
        "llm_generation_max_concurrent",
        "llm_auxiliary_max_concurrent",
        "llm_request_timeout",
        "llm_auxiliary_timeout",
        "embedding_max_concurrent",
        "ingestion_max_concurrent",
        "reranker_max_concurrent",
        "bm25_search_max_concurrent",
        "semaphore_acquire_timeout",
        "benchmark_max_concurrent",
        "benchmark_judge_max_concurrent",
        "benchmark_judge_grouped_enabled",
        "benchmark_judge_initial_tokens",
        "benchmark_judge_retry_tokens",
        "benchmark_judge_model",
        "tei_embed_model",
        "embedding_revision",
        "document_domain_marker_threshold",
        "embed_dim",
        "curator_scope_max_ids",
        "redis_host",
        "redis_port",
        "redis_user",
        "redis_db",
        "worker_max_concurrent",
        "redis_socket_timeout",
        "redis_max_connections",
        "file_backend",
        "s3_endpoint",
        "s3_bucket",
        "s3_region",
        "s3_max_concurrent",
        "supported_extensions",
        "data_dir",
        "uploads_prefix",
        "allowed_origins",
        "forwarded_allow_ips",
        "jwt_algorithm",
        "jwt_expire_minutes",
        "admin_email",
        "timezone",
        "stage",
        "rate_limit_redis_prefix",
        "rate_limit_max_buckets",
        "llm_breaker_fail_max",
        "llm_breaker_timeout_duration",
    }
)

METADATA_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "version",
        "service_start_datetime",
    }
)

TARGET_DYNAMIC_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "qdrant_url",
        "collection_name",
        "ollama_base_url",
        "llm_provider",
        "llm_model",
        "llm_temperature",
        "llm_top_p",
        "llm_num_ctx_narrow",
        "llm_num_ctx_broad",
        "llm_num_predict_narrow",
        "llm_num_predict_broad",
        "openrouter_base_url",
        "openrouter_model",
        "ml_provider",
        "tei_embed_url",
        "tei_rerank_url",
        "deepinfra_base_url",
        "deepinfra_embed_model",
        "deepinfra_rerank_model",
        "embedding_http_max_connections",
        "rerank_http_max_connections",
        "embedding_http_max_keepalive_connections",
        "rerank_http_max_keepalive_connections",
        "ocr_engine",
        "ocr_enabled",
        "ocr_lang_paddle",
        "ocr_lang_surya",
        "ocr_dpi",
        "ocr_min_chars",
        "retriever_fetch_k",
        "retriever_top_k",
        "retriever_fetch_k_broad",
        "retriever_top_k_broad",
        "history_window",
        "rerank_min_score",
        "rerank_score_gap_ratio",
        "source_min_score",
        "citation_filter_enabled",
        "exact_ref_sparse_boost",
        "chunk_size",
        "chunk_overlap",
        "legal_chunk_size",
        "legal_chunk_overlap",
        "decree_chunk_size",
        "decree_chunk_overlap",
        "embed_batch_size",
        "hybrid_enabled",
        "bm25_fetch_k",
        "rrf_k",
        "dense_weight",
        "sparse_weight",
        "relevance_gate_enabled",
        "condense_enabled",
        "decomposition_enabled",
        "rolling_summary_enabled",
        "cache_enabled",
        "pii_redaction_enabled",
        "max_upload_size_mb",
        "job_cleanup_days",
        "stuck_job_timeout_minutes",
        "stale_pending_timeout_minutes",
        "rate_limit_enabled",
        "rate_limit_fail_open",
        "rate_limit_login_per_minute",
        "rate_limit_chat_per_minute",
        "rate_limit_chat_per_day",
        "rate_limit_upload_per_hour",
        "rate_limit_search_per_minute",
        "rate_limit_write_per_minute",
        "rate_limit_benchmark_per_hour",
        "cost_rate_limit_enabled",
        "cost_hourly_limit",
        "cost_daily_limit",
    }
)

# Legacy names are recognized only by the offline migration parser.
LEGACY_POOL_SETTING_KEYS: frozenset[str] = frozenset({"http_pool_max_connections", "http_pool_max_keepalive"})


@dataclass(frozen=True)
class DynamicParam:
    """Declaration of a single hot-reloadable config parameter."""

    key: str
    type: type
    category: str
    description: str
    min_val: float | None = None
    max_val: float | None = None
    allowed: list[str] | None = field(default=None, repr=False)
    domain_key: str | None = None


# nginx allows 64 MiB per request; reserve 1 MiB for multipart metadata.
UPLOAD_REQUEST_CEILING_MB = 64
MAX_UPLOAD_FILE_MB = UPLOAD_REQUEST_CEILING_MB - 1

DYNAMIC_PARAMS: list[DynamicParam] = [
    DynamicParam("qdrant_url", str, "corpus", "Qdrant endpoint; coordinated corpus switch only"),
    DynamicParam(
        "collection_name", str, "corpus", "Alias/physical collection; coordinated corpus switch only"
    ),
    DynamicParam(
        "cost_rate_limit_enabled", bool, "cost", "USD per principal; unsupported paid calls blocked"
    ),
    DynamicParam(
        "cost_hourly_limit", float, "cost", "Rolling hour USD; includes pending reservations", 0.000000001
    ),
    DynamicParam(
        "cost_daily_limit", float, "cost", "Rolling day USD; includes pending reservations", 0.000000001
    ),
    DynamicParam(
        "max_upload_size_mb",
        int,
        "ingestion",
        "File limit MiB; nginx request ceiling 64 MiB including multipart",
        1,
        MAX_UPLOAD_FILE_MB,
    ),
    DynamicParam(
        "job_cleanup_days",
        int,
        "jobs",
        "Retention days; reducing deletes older completed/failed jobs on next hourly cleanup",
        1,
    ),
    DynamicParam("ollama_base_url", str, "llm", "Ollama endpoint (deployment-approved origin)"),
    DynamicParam("openrouter_base_url", str, "openrouter", "OpenRouter endpoint (canonical HTTPS origin)"),
    DynamicParam("tei_embed_url", str, "ml", "TEI embedding endpoint (same model identity required)"),
    DynamicParam("tei_rerank_url", str, "ml", "TEI reranker endpoint (deployment-approved origin)"),
    DynamicParam("deepinfra_base_url", str, "ml", "DeepInfra endpoint (canonical HTTPS origin)"),
    DynamicParam("embedding_http_max_connections", int, "ml", "Embedding HTTP connections per process", 1),
    DynamicParam("embedding_http_max_keepalive_connections", int, "ml", "Embedding idle HTTP connections", 0),
    DynamicParam("rerank_http_max_connections", int, "ml", "Reranker HTTP connections per process", 1),
    DynamicParam("rerank_http_max_keepalive_connections", int, "ml", "Reranker idle HTTP connections", 0),
    # ── RAG ─────────────────────────────────────────────────────
    DynamicParam("retriever_fetch_k", int, "rag", "Retriever fetch count (narrow)", 1, 200),
    DynamicParam("retriever_top_k", int, "rag", "Retriever top-k (narrow)", 1, 50),
    DynamicParam("retriever_fetch_k_broad", int, "rag", "Retriever fetch count (broad)", 1, 200),
    DynamicParam("retriever_top_k_broad", int, "rag", "Retriever top-k (broad)", 1, 50),
    DynamicParam("history_window", int, "rag", "Chat history window", 0, 50),
    DynamicParam("chunk_size", int, "rag", "Document chunk size (chars)", 100, 5000),
    DynamicParam("chunk_overlap", int, "rag", "Chunk overlap (chars)", 0, 1000),
    DynamicParam(
        "legal_chunk_size",
        int,
        "rag",
        "Legal doc chunk size (chars)",
        100,
        5000,
        domain_key=DocDomain.LEGAL.value,
    ),
    DynamicParam(
        "legal_chunk_overlap",
        int,
        "rag",
        "Legal doc chunk overlap (chars)",
        0,
        1000,
        domain_key=DocDomain.LEGAL.value,
    ),
    DynamicParam(
        "decree_chunk_size",
        int,
        "rag",
        "Decree doc chunk size (chars)",
        100,
        5000,
        domain_key=DocDomain.DECREE.value,
    ),
    DynamicParam(
        "decree_chunk_overlap",
        int,
        "rag",
        "Decree doc chunk overlap (chars)",
        0,
        1000,
        domain_key=DocDomain.DECREE.value,
    ),
    DynamicParam("source_min_score", float, "rag", "Min source relevance score", 0.0, 1.0),
    # ── Hybrid search ───────────────────────────────────────────
    DynamicParam("hybrid_enabled", bool, "hybrid", "Enable hybrid search (dense+sparse)"),
    DynamicParam("bm25_fetch_k", int, "hybrid", "BM25 fetch count", 1, 200),
    DynamicParam("rrf_k", int, "hybrid", "RRF fusion parameter", 1, 200),
    DynamicParam("dense_weight", float, "hybrid", "Dense vector weight", 0.0, 10.0),
    DynamicParam("sparse_weight", float, "hybrid", "Sparse vector weight", 0.0, 10.0),
    # ── Reranker ────────────────────────────────────────────────
    DynamicParam("rerank_min_score", float, "reranker", "Min reranker score threshold", 0.0, 1.0),
    DynamicParam("rerank_score_gap_ratio", float, "reranker", "Reranker score gap ratio", 0.0, 1.0),
    DynamicParam("citation_filter_enabled", bool, "reranker", "Enable citation filter"),
    DynamicParam("exact_ref_sparse_boost", float, "reranker", "Exact ref sparse boost", 0.0, 10.0),
    # ── Ingestion ───────────────────────────────────────────────
    DynamicParam("embed_batch_size", int, "ingestion", "Embedding batch size", 1, 128),
    # ── Feature toggles ─────────────────────────────────────────
    DynamicParam("relevance_gate_enabled", bool, "toggles", "Enable relevance gate"),
    DynamicParam("condense_enabled", bool, "toggles", "Enable follow-up condensing"),
    DynamicParam("decomposition_enabled", bool, "toggles", "Enable query decomposition"),
    DynamicParam("rolling_summary_enabled", bool, "toggles", "Enable rolling summary"),
    DynamicParam("cache_enabled", bool, "toggles", "Enable response cache"),
    DynamicParam("pii_redaction_enabled", bool, "toggles", "Enable PII redaction"),
    # ── LLM ─────────────────────────────────────────────────────
    DynamicParam(
        "llm_provider", str, "llm", "LLM provider", allowed=[provider.value for provider in LLMProvider]
    ),
    DynamicParam("llm_model", str, "llm", "LLM model name"),
    DynamicParam("llm_temperature", float, "llm", "LLM temperature", 0.0, 2.0),
    DynamicParam("llm_top_p", float, "llm", "LLM top-p", 0.0, 1.0),
    DynamicParam("llm_num_ctx_narrow", int, "llm", "LLM context window (narrow)", 512, 131072),
    DynamicParam("llm_num_ctx_broad", int, "llm", "LLM context window (broad)", 512, 131072),
    DynamicParam("llm_num_predict_narrow", int, "llm", "LLM max tokens predict (narrow)", 64, 16384),
    DynamicParam("llm_num_predict_broad", int, "llm", "LLM max tokens predict (broad)", 64, 16384),
    # ── OpenRouter ──────────────────────────────────────────────
    DynamicParam("openrouter_model", str, "openrouter", "OpenRouter model name"),
    # ── ML Provider ─────────────────────────────────────────────
    DynamicParam(
        "ml_provider",
        str,
        "ml",
        "ML provider for embedding/reranking",
        allowed=["tei", "deepinfra"],
    ),
    DynamicParam("deepinfra_embed_model", str, "ml", "DeepInfra embedding model"),
    DynamicParam("deepinfra_rerank_model", str, "ml", "DeepInfra reranker model"),
    # ── OCR ─────────────────────────────────────────────────────
    DynamicParam("ocr_enabled", bool, "ocr", "Enable OCR processing"),
    DynamicParam("ocr_engine", str, "ocr", "OCR engine", allowed=["paddleocr", "surya"]),
    DynamicParam("ocr_dpi", int, "ocr", "OCR DPI for scanned pages", 72, 600),
    DynamicParam("ocr_min_chars", int, "ocr", "Min chars to consider page text", 0, 500),
    DynamicParam("ocr_lang_surya", list, "ocr", "Surya OCR languages"),
    DynamicParam("ocr_lang_paddle", str, "ocr", "PaddleOCR language"),
    DynamicParam("rate_limit_enabled", bool, "rate_limit", "Enable request rate limits"),
    DynamicParam("rate_limit_fail_open", bool, "rate_limit", "Allow requests when Redis is unavailable"),
    DynamicParam("rate_limit_login_per_minute", int, "rate_limit", "login per minute; rolling window", 1),
    DynamicParam("rate_limit_chat_per_minute", int, "rate_limit", "chat per minute; rolling window", 1),
    DynamicParam("rate_limit_chat_per_day", int, "rate_limit", "chat per day; rolling window", 1),
    DynamicParam("rate_limit_upload_per_hour", int, "rate_limit", "upload per hour; rolling window", 1),
    DynamicParam("rate_limit_search_per_minute", int, "rate_limit", "search per minute; rolling window", 1),
    DynamicParam("rate_limit_write_per_minute", int, "rate_limit", "write per minute; rolling window", 1),
    DynamicParam("rate_limit_benchmark_per_hour", int, "rate_limit", "benchmark per hour; rolling window", 1),
    # ── Background jobs ─────────────────────────────────────────
    DynamicParam("stuck_job_timeout_minutes", int, "jobs", "Minutes before PROCESSING doc is stuck", 5, 120),
    DynamicParam("stale_pending_timeout_minutes", int, "jobs", "Minutes before PENDING job is stale", 5, 120),
]

# Derived lookup: key -> (settings_attr, python_type) — global params only
DYNAMIC_FIELDS_MAP: dict[str, tuple[str, type]] = {
    p.key: (p.key, p.type) for p in DYNAMIC_PARAMS if not p.domain_key
}
