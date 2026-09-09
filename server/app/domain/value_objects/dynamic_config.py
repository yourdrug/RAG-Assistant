"""Declarative dynamic config parameters — single source of truth.

Every hot-reloadable config parameter is declared once here. The
``_build_defaults()`` seed function and the ``_DYNAMIC_FIELDS`` mapping
are both derived from this list, eliminating the risk of triple-sync bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


DYNAMIC_PARAMS: list[DynamicParam] = [
    # ── RAG ─────────────────────────────────────────────────────
    DynamicParam("retriever_fetch_k", int, "rag", "Retriever fetch count (narrow)", 1, 200),
    DynamicParam("retriever_top_k", int, "rag", "Retriever top-k (narrow)", 1, 50),
    DynamicParam("retriever_fetch_k_broad", int, "rag", "Retriever fetch count (broad)", 1, 200),
    DynamicParam("retriever_top_k_broad", int, "rag", "Retriever top-k (broad)", 1, 50),
    DynamicParam("history_window", int, "rag", "Chat history window", 0, 50),
    DynamicParam("chunk_size", int, "rag", "Document chunk size (chars)", 100, 5000),
    DynamicParam("chunk_overlap", int, "rag", "Chunk overlap (chars)", 0, 1000),
    DynamicParam(
        "legal_chunk_size", int, "rag", "Legal doc chunk size (chars)", 100, 5000, domain_key="legal",
    ),
    DynamicParam(
        "legal_chunk_overlap", int, "rag", "Legal doc chunk overlap (chars)", 0, 1000, domain_key="legal",
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
    DynamicParam("llm_provider", str, "llm", "LLM provider", allowed=["ollama", "openrouter"]),
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
        "ml_provider", str, "ml", "ML provider for embedding/reranking",
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
    # ── Storage ─────────────────────────────────────────────────
    DynamicParam("s3_endpoint", str, "storage", "S3 endpoint URL"),
    DynamicParam("s3_bucket", str, "storage", "S3 bucket name"),
    DynamicParam("s3_region", str, "storage", "S3 region"),
    DynamicParam("s3_access_key", str, "storage", "S3 access key"),
    DynamicParam("s3_secret_key", str, "storage", "S3 secret key"),
    # ── Background jobs ─────────────────────────────────────────
    DynamicParam("stuck_job_timeout_minutes", int, "jobs", "Minutes before PROCESSING doc is stuck", 5, 120),
    DynamicParam("stale_pending_timeout_minutes", int, "jobs", "Minutes before PENDING job is stale", 5, 120),
]

# Derived lookup: key -> (settings_attr, python_type) — global params only
DYNAMIC_FIELDS_MAP: dict[str, tuple[str, type]] = {
    p.key: (p.key, p.type) for p in DYNAMIC_PARAMS if not p.domain_key
}
