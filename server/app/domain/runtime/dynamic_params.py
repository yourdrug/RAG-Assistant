"""Declarative dynamic config parameters — single source of truth.

Every hot-reloadable config parameter is declared once here. The
``_build_defaults()`` seed function and the ``_DYNAMIC_FIELDS`` mapping
are both derived from this list, eliminating the risk of triple-sync bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from domain.knowledge.enums import DocDomain
from domain.runtime.enums import LLMProvider, MLProvider


class DynamicParamCategory(StrEnum):
    CORPUS = "corpus"
    COST = "cost"
    HYBRID = "hybrid"
    INGESTION = "ingestion"
    JOBS = "jobs"
    LLM = "llm"
    ML = "ml"
    OCR = "ocr"
    OPENROUTER = "openrouter"
    RAG = "rag"
    RATE_LIMIT = "rate_limit"
    RERANKER = "reranker"
    TOGGLES = "toggles"


@dataclass(frozen=True)
class DynamicParam:
    """Declaration of a single hot-reloadable config parameter."""

    key: str
    type: type
    category: DynamicParamCategory
    description: str
    min_val: float | None = None
    max_val: float | None = None
    allowed: list[str] | None = field(default=None, repr=False)
    domain_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", DynamicParamCategory(self.category))


# nginx allows 64 MiB per request; reserve 1 MiB for multipart metadata.
UPLOAD_REQUEST_CEILING_MB = 64
MAX_UPLOAD_FILE_MB = UPLOAD_REQUEST_CEILING_MB - 1

DYNAMIC_PARAMS: list[DynamicParam] = [
    DynamicParam(
        "qdrant_url", str, DynamicParamCategory.CORPUS, "Qdrant endpoint; coordinated corpus switch only"
    ),
    DynamicParam(
        "collection_name",
        str,
        DynamicParamCategory.CORPUS,
        "Alias/physical collection; coordinated corpus switch only",
    ),
    DynamicParam(
        "cost_rate_limit_enabled",
        bool,
        DynamicParamCategory.COST,
        "USD per principal; unsupported paid calls blocked",
    ),
    DynamicParam(
        "cost_hourly_limit",
        float,
        DynamicParamCategory.COST,
        "Rolling hour USD; includes pending reservations",
        0.000000001,
    ),
    DynamicParam(
        "cost_daily_limit",
        float,
        DynamicParamCategory.COST,
        "Rolling day USD; includes pending reservations",
        0.000000001,
    ),
    DynamicParam(
        "max_upload_size_mb",
        int,
        DynamicParamCategory.INGESTION,
        "File limit MiB; nginx request ceiling 64 MiB including multipart",
        1,
        MAX_UPLOAD_FILE_MB,
    ),
    DynamicParam(
        "job_cleanup_days",
        int,
        DynamicParamCategory.JOBS,
        "Retention days; reducing deletes older completed/failed jobs on next hourly cleanup",
        1,
    ),
    DynamicParam(
        "ollama_base_url", str, DynamicParamCategory.LLM, "Ollama endpoint (deployment-approved origin)"
    ),
    DynamicParam(
        "openrouter_base_url",
        str,
        DynamicParamCategory.OPENROUTER,
        "OpenRouter endpoint (canonical HTTPS origin)",
    ),
    DynamicParam(
        "tei_embed_url", str, DynamicParamCategory.ML, "TEI embedding endpoint (same model identity required)"
    ),
    DynamicParam(
        "tei_rerank_url", str, DynamicParamCategory.ML, "TEI reranker endpoint (deployment-approved origin)"
    ),
    DynamicParam(
        "deepinfra_base_url", str, DynamicParamCategory.ML, "DeepInfra endpoint (canonical HTTPS origin)"
    ),
    DynamicParam(
        "embedding_http_max_connections",
        int,
        DynamicParamCategory.ML,
        "Embedding HTTP connections per process",
        1,
    ),
    DynamicParam(
        "embedding_http_max_keepalive_connections",
        int,
        DynamicParamCategory.ML,
        "Embedding idle HTTP connections",
        0,
    ),
    DynamicParam(
        "rerank_http_max_connections",
        int,
        DynamicParamCategory.ML,
        "Reranker HTTP connections per process",
        1,
    ),
    DynamicParam(
        "rerank_http_max_keepalive_connections",
        int,
        DynamicParamCategory.ML,
        "Reranker idle HTTP connections",
        0,
    ),
    # ── RAG ─────────────────────────────────────────────────────
    DynamicParam(
        "retriever_fetch_k", int, DynamicParamCategory.RAG, "Retriever fetch count (narrow)", 1, 200
    ),
    DynamicParam("retriever_top_k", int, DynamicParamCategory.RAG, "Retriever top-k (narrow)", 1, 50),
    DynamicParam(
        "retriever_fetch_k_broad", int, DynamicParamCategory.RAG, "Retriever fetch count (broad)", 1, 200
    ),
    DynamicParam("retriever_top_k_broad", int, DynamicParamCategory.RAG, "Retriever top-k (broad)", 1, 50),
    DynamicParam("history_window", int, DynamicParamCategory.RAG, "Chat history window", 0, 50),
    DynamicParam("chunk_size", int, DynamicParamCategory.RAG, "Document chunk size (chars)", 100, 5000),
    DynamicParam("chunk_overlap", int, DynamicParamCategory.RAG, "Chunk overlap (chars)", 0, 1000),
    DynamicParam(
        "legal_chunk_size",
        int,
        DynamicParamCategory.RAG,
        "Legal doc chunk size (chars)",
        100,
        5000,
        domain_key=DocDomain.LEGAL.value,
    ),
    DynamicParam(
        "legal_chunk_overlap",
        int,
        DynamicParamCategory.RAG,
        "Legal doc chunk overlap (chars)",
        0,
        1000,
        domain_key=DocDomain.LEGAL.value,
    ),
    DynamicParam(
        "decree_chunk_size",
        int,
        DynamicParamCategory.RAG,
        "Decree doc chunk size (chars)",
        100,
        5000,
        domain_key=DocDomain.DECREE.value,
    ),
    DynamicParam(
        "decree_chunk_overlap",
        int,
        DynamicParamCategory.RAG,
        "Decree doc chunk overlap (chars)",
        0,
        1000,
        domain_key=DocDomain.DECREE.value,
    ),
    DynamicParam("source_min_score", float, DynamicParamCategory.RAG, "Min source relevance score", 0.0, 1.0),
    # ── Hybrid search ───────────────────────────────────────────
    DynamicParam("hybrid_enabled", bool, DynamicParamCategory.HYBRID, "Enable hybrid search (dense+sparse)"),
    DynamicParam("bm25_fetch_k", int, DynamicParamCategory.HYBRID, "BM25 fetch count", 1, 200),
    DynamicParam("rrf_k", int, DynamicParamCategory.HYBRID, "RRF fusion parameter", 1, 200),
    DynamicParam("dense_weight", float, DynamicParamCategory.HYBRID, "Dense vector weight", 0.0, 10.0),
    DynamicParam("sparse_weight", float, DynamicParamCategory.HYBRID, "Sparse vector weight", 0.0, 10.0),
    # ── Reranker ────────────────────────────────────────────────
    DynamicParam(
        "rerank_min_score", float, DynamicParamCategory.RERANKER, "Min reranker score threshold", 0.0, 1.0
    ),
    DynamicParam(
        "rerank_score_gap_ratio", float, DynamicParamCategory.RERANKER, "Reranker score gap ratio", 0.0, 1.0
    ),
    DynamicParam("citation_filter_enabled", bool, DynamicParamCategory.RERANKER, "Enable citation filter"),
    DynamicParam(
        "exact_ref_sparse_boost", float, DynamicParamCategory.RERANKER, "Exact ref sparse boost", 0.0, 10.0
    ),
    # ── Ingestion ───────────────────────────────────────────────
    DynamicParam("embed_batch_size", int, DynamicParamCategory.INGESTION, "Embedding batch size", 1, 128),
    # ── Feature toggles ─────────────────────────────────────────
    DynamicParam("relevance_gate_enabled", bool, DynamicParamCategory.TOGGLES, "Enable relevance gate"),
    DynamicParam("condense_enabled", bool, DynamicParamCategory.TOGGLES, "Enable follow-up condensing"),
    DynamicParam("decomposition_enabled", bool, DynamicParamCategory.TOGGLES, "Enable query decomposition"),
    DynamicParam("rolling_summary_enabled", bool, DynamicParamCategory.TOGGLES, "Enable rolling summary"),
    DynamicParam("cache_enabled", bool, DynamicParamCategory.TOGGLES, "Enable response cache"),
    DynamicParam("pii_redaction_enabled", bool, DynamicParamCategory.TOGGLES, "Enable PII redaction"),
    # ── LLM ─────────────────────────────────────────────────────
    DynamicParam(
        "llm_provider",
        str,
        DynamicParamCategory.LLM,
        "LLM provider",
        allowed=[provider.value for provider in LLMProvider],
    ),
    DynamicParam("llm_model", str, DynamicParamCategory.LLM, "LLM model name"),
    DynamicParam("llm_temperature", float, DynamicParamCategory.LLM, "LLM temperature", 0.0, 2.0),
    DynamicParam("llm_top_p", float, DynamicParamCategory.LLM, "LLM top-p", 0.0, 1.0),
    DynamicParam(
        "llm_num_ctx_narrow", int, DynamicParamCategory.LLM, "LLM context window (narrow)", 512, 131072
    ),
    DynamicParam(
        "llm_num_ctx_broad", int, DynamicParamCategory.LLM, "LLM context window (broad)", 512, 131072
    ),
    DynamicParam(
        "llm_num_predict_narrow", int, DynamicParamCategory.LLM, "LLM max tokens predict (narrow)", 64, 16384
    ),
    DynamicParam(
        "llm_num_predict_broad", int, DynamicParamCategory.LLM, "LLM max tokens predict (broad)", 64, 16384
    ),
    # ── OpenRouter ──────────────────────────────────────────────
    DynamicParam("openrouter_model", str, DynamicParamCategory.OPENROUTER, "OpenRouter model name"),
    # ── ML Provider ─────────────────────────────────────────────
    DynamicParam(
        "ml_provider",
        str,
        DynamicParamCategory.ML,
        "ML provider for embedding/reranking",
        allowed=[provider.value for provider in MLProvider],
    ),
    DynamicParam("deepinfra_embed_model", str, DynamicParamCategory.ML, "DeepInfra embedding model"),
    DynamicParam("deepinfra_rerank_model", str, DynamicParamCategory.ML, "DeepInfra reranker model"),
    # ── OCR ─────────────────────────────────────────────────────
    DynamicParam("ocr_enabled", bool, DynamicParamCategory.OCR, "Enable OCR processing"),
    DynamicParam("ocr_engine", str, DynamicParamCategory.OCR, "OCR engine", allowed=["paddleocr", "surya"]),
    DynamicParam("ocr_dpi", int, DynamicParamCategory.OCR, "OCR DPI for scanned pages", 72, 600),
    DynamicParam("ocr_min_chars", int, DynamicParamCategory.OCR, "Min chars to consider page text", 0, 500),
    DynamicParam("ocr_lang_surya", list, DynamicParamCategory.OCR, "Surya OCR languages"),
    DynamicParam("ocr_lang_paddle", str, DynamicParamCategory.OCR, "PaddleOCR language"),
    DynamicParam("rate_limit_enabled", bool, DynamicParamCategory.RATE_LIMIT, "Enable request rate limits"),
    DynamicParam(
        "rate_limit_fail_open",
        bool,
        DynamicParamCategory.RATE_LIMIT,
        "Allow requests when Redis is unavailable",
    ),
    DynamicParam(
        "rate_limit_login_per_minute",
        int,
        DynamicParamCategory.RATE_LIMIT,
        "login per minute; rolling window",
        1,
    ),
    DynamicParam(
        "rate_limit_chat_per_minute",
        int,
        DynamicParamCategory.RATE_LIMIT,
        "chat per minute; rolling window",
        1,
    ),
    DynamicParam(
        "rate_limit_chat_per_day", int, DynamicParamCategory.RATE_LIMIT, "chat per day; rolling window", 1
    ),
    DynamicParam(
        "rate_limit_upload_per_hour",
        int,
        DynamicParamCategory.RATE_LIMIT,
        "upload per hour; rolling window",
        1,
    ),
    DynamicParam(
        "rate_limit_search_per_minute",
        int,
        DynamicParamCategory.RATE_LIMIT,
        "search per minute; rolling window",
        1,
    ),
    DynamicParam(
        "rate_limit_write_per_minute",
        int,
        DynamicParamCategory.RATE_LIMIT,
        "write per minute; rolling window",
        1,
    ),
    DynamicParam(
        "rate_limit_benchmark_per_hour",
        int,
        DynamicParamCategory.RATE_LIMIT,
        "benchmark per hour; rolling window",
        1,
    ),
    # ── Background jobs ─────────────────────────────────────────
    DynamicParam(
        "stuck_job_timeout_minutes",
        int,
        DynamicParamCategory.JOBS,
        "Minutes before PROCESSING doc is stuck",
        5,
        120,
    ),
    DynamicParam(
        "stale_pending_timeout_minutes",
        int,
        DynamicParamCategory.JOBS,
        "Minutes before PENDING job is stale",
        5,
        120,
    ),
]

# Derived lookup: key -> (settings_attr, python_type) — global params only
DYNAMIC_FIELDS_MAP: dict[str, tuple[str, type]] = {
    p.key: (p.key, p.type) for p in DYNAMIC_PARAMS if not p.domain_key
}
