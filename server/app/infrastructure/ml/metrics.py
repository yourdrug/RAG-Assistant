"""Prometheus metrics and collectors for the RAG pipeline and infrastructure.

Defines counters, histograms, and gauges for RAG query quality, document
ingestion throughput, and infrastructure health (Postgres pool, Qdrant
collection size, Ollama GPU usage).  The infrastructure gauges are refreshed
periodically by the Scheduler (``infrastructure.scheduler``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import httpx
from config import settings
from prometheus_client import Counter, Gauge, Histogram
from sqlalchemy.pool import QueuePool

from infrastructure.database.database import database

if TYPE_CHECKING:
    from infrastructure.ml.client_registry import MLClientRegistry

log = logging.getLogger("default")

# ---------------------------------------------------------------------------
# RAG Pipeline metrics
# ---------------------------------------------------------------------------

RAG_STAGE_DURATION = Histogram(
    "rag_stage_duration_seconds",
    "Latency of individual RAG pipeline stages",
    ["stage"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    ["breadth", "answer_type"],
)

RAG_ANSWER_LENGTH = Histogram(
    "rag_answer_length_chars",
    "Length of LLM answer in characters",
    buckets=(10, 50, 100, 200, 500, 1000, 2000, 5000),
)

RAG_RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Number of chunks returned by retriever (after rerank)",
    ["breadth"],
    buckets=(1, 2, 3, 5, 8, 10, 15, 20),
)

RAG_SIMILARITY_SCORES = Histogram(
    "rag_similarity_scores",
    "Average similarity score from retrieval",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

RAG_BREADTH_TOTAL = Counter(
    "rag_breadth_total",
    "Question breadth classification count",
    ["breadth"],
)

RAG_NOT_FOUND_TOTAL = Counter(
    "rag_not_found_total",
    "Number of answers indicating info not found in documents",
)

RAG_DECOMPOSED_TOTAL = Counter(
    "rag_decomposed_queries_total",
    "Number of queries that were decomposed into sub-queries",
)

RAG_RELEVANCE_GATE_TOTAL = Counter(
    "rag_relevance_gate_total",
    "Relevance gate check results",
    ["result"],  # "passed" | "rejected"
)

RAG_CACHE_HITS_TOTAL = Counter(
    "rag_cache_hit_total",
    "Semantic answer cache hits",
)

RAG_CACHE_MISSES_TOTAL = Counter(
    "rag_cache_miss_total",
    "Semantic answer cache misses",
)

RAG_SELF_RAG_RETRIES = Counter(
    "rag_self_rag_retries_total",
    "Self-RAG retry attempts (insufficient context → refined query)",
)

# ---------------------------------------------------------------------------
# Ingestion metrics
# ---------------------------------------------------------------------------

INGEST_DOCUMENTS_TOTAL = Counter(
    "ingest_documents_total",
    "Documents processed via background task",
    ["status"],
)

INGEST_DOCUMENT_DURATION = Histogram(
    "ingest_document_duration_seconds",
    "Time to process a single document (parse + split + upload)",
    ["status"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

INGEST_CHUNKS_TOTAL = Counter(
    "ingest_chunks_total",
    "Total chunks created during ingestion",
)

INGEST_FILES_TOTAL = Counter(
    "ingest_files_total",
    "Files processed during full ingestion",
    ["status"],
)

INGEST_PDF_PAGES_TOTAL = Counter(
    "ingest_pdf_pages_total",
    "PDF page quality classification",
    ["quality"],
)

INGEST_PDF_BAD_RATIO = Histogram(
    "ingest_pdf_bad_ratio",
    "Ratio of bad pages (missing + garbled) in PDF",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
)

# ---------------------------------------------------------------------------
# Infrastructure gauges (updated by periodic collector)
# ---------------------------------------------------------------------------

DB_POOL_IN_USE = Gauge(
    "db_pool_connections_in_use",
    "Postgres connections currently checked out",
)

DB_POOL_IDLE = Gauge(
    "db_pool_connections_idle",
    "Postgres connections idle in pool",
)

DB_POOL_OVERFLOW = Gauge(
    "db_pool_overflow",
    "Postgres overflow connections beyond pool_size",
)

QDRANT_POINTS = Gauge(
    "qdrant_collection_points",
    "Number of points in Qdrant collection",
)

BM25_INDEX_SIZE = Gauge(
    "bm25_index_size",
    "Number of documents in BM25 index",
)

OLLAMA_GPU_MEMORY_BYTES = Gauge(
    "ollama_gpu_memory_bytes",
    "Ollama GPU memory usage in bytes",
    ["model"],
)

OLLAMA_RAM_MEMORY_BYTES = Gauge(
    "ollama_ram_memory_bytes",
    "Ollama RAM memory usage in bytes",
    ["model"],
)

# ---------------------------------------------------------------------------
# Config listener metrics
# ---------------------------------------------------------------------------

CONFIG_LISTENER_CONNECTED = Gauge(
    "config_listener_connected",
    "1 if Postgres LISTEN/NOTIFY connection for dynamic config is active, else 0",
)

CONFIG_RESYNC_TOTAL = Counter(
    "config_resync_total",
    "Number of full config resyncs performed",
    ["trigger"],  # "reconnect" | "periodic" | "manual"
)

CONFIG_NOTIFY_RECEIVED_TOTAL = Counter(
    "config_notify_received_total",
    "Number of config_changed NOTIFY payloads received",
)

# ---------------------------------------------------------------------------
# HTTP request metrics (used by MetricsMiddleware)
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["handler", "method", "status"],
)

# ---------------------------------------------------------------------------
# GenAI observability (OTel semantic conventions)
# ---------------------------------------------------------------------------

LLM_TOKEN_USAGE = Counter(
    "rag_llm_tokens_total",
    "LLM token usage (input + output)",
    ["model", "direction", "operation"],
    # direction: "input" | "output"
    # operation: "generate" | "condense" | "decompose" | "relevance_gate" | "judge"
)


def record_llm_usage(
    model: str,
    operation: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Record LLM token usage.

    Extracts from LangChain response.usage_metadata when available.
    """
    if input_tokens is not None and input_tokens > 0:
        LLM_TOKEN_USAGE.labels(model=model, direction="input", operation=operation).inc(input_tokens)
    if output_tokens is not None and output_tokens > 0:
        LLM_TOKEN_USAGE.labels(model=model, direction="output", operation=operation).inc(output_tokens)


def extract_usage_from_langchain(response) -> tuple[int | None, int | None]:
    """Extract (input_tokens, output_tokens) from a LangChain AIMessage or chunk.

    Supports both usage_metadata (LangChain v0.3+) and response_metadata.
    Returns (None, None) if not available.
    """
    # Try usage_metadata first (LangChain >=0.3)
    usage = getattr(response, "usage_metadata", None)
    if usage and isinstance(usage, dict):
        return usage.get("input_tokens"), usage.get("output_tokens")

    # Try response_metadata (older LangChain / OpenAI format)
    meta = getattr(response, "response_metadata", None)
    if meta and isinstance(meta, dict):
        usage = meta.get("usage", {})
        if usage:
            return usage.get("prompt_tokens"), usage.get("completion_tokens")

    return None, None


# ---------------------------------------------------------------------------
# Helper: record RAG pipeline answer metrics (called after generation)
# ---------------------------------------------------------------------------

_NOT_FOUND_PATTERNS = (
    "не найден",
    "не найдена",
    "не найдено",
    "нет информации",
    "не удалось найти",
    "информация не найдена",
    "в предоставленных документах",
    "в документах нет",
    "не обнаружен",
)


def record_rag_answer(
    breadth: str,
    answer: str,
    retrieved_count: int,
    avg_similarity: float,
) -> None:
    """Record answer-level metrics after RAG generation completes."""
    RAG_RETRIEVED_CHUNKS.labels(breadth=breadth).observe(retrieved_count)
    RAG_SIMILARITY_SCORES.observe(avg_similarity)
    RAG_ANSWER_LENGTH.observe(len(answer))

    lower = answer.lower()
    is_not_found = any(p in lower for p in _NOT_FOUND_PATTERNS)
    answer_type = "not_found" if is_not_found else "found"
    RAG_QUERIES_TOTAL.labels(breadth=breadth, answer_type=answer_type).inc()
    if is_not_found:
        RAG_NOT_FOUND_TOTAL.inc()


# ---------------------------------------------------------------------------
# Periodic infrastructure collector
# ---------------------------------------------------------------------------


async def _collect_db_pool_metrics() -> None:
    engine = database.master_node.async_engine if database.master_node else None
    if engine is None:
        return
    pool = cast(QueuePool, engine.pool)
    DB_POOL_IN_USE.set(pool.checkedout())
    DB_POOL_IDLE.set(pool.checkedin())
    DB_POOL_OVERFLOW.set(pool.overflow())


async def _collect_qdrant_metrics(ml_clients: MLClientRegistry | None) -> None:
    if ml_clients is not None:
        client = ml_clients.qdrant_client()
    else:
        from infrastructure.ml.factories import create_qdrant_client

        client = create_qdrant_client()
    info = client.get_collection(settings.collection_name)
    QDRANT_POINTS.set(info.points_count or 0)


async def _collect_bm25_metrics(ml_clients: MLClientRegistry | None) -> None:
    if ml_clients is not None:
        bm25 = ml_clients.bm25_index()
    else:
        from infrastructure.ml.factories import load_bm25_index

        bm25 = load_bm25_index()
    if bm25 is not None:
        BM25_INDEX_SIZE.set(len(bm25.hashes))


async def _collect_ollama_metrics() -> None:
    async with httpx.AsyncClient(timeout=3) as http:
        r = await http.get(f"{settings.ollama_base_url}/api/tags")
        if r.status_code == 200:
            data = r.json()
            for model in data.get("models", []):
                model_name = model.get("name", "unknown")
                model_size = model.get("size", 0)
                OLLAMA_GPU_MEMORY_BYTES.labels(model=model_name).set(model_size)
                OLLAMA_RAM_MEMORY_BYTES.labels(model=model_name).set(model_size)


async def collect_infra_metrics(ml_clients: MLClientRegistry | None = None) -> None:
    """Update infrastructure gauges. Called periodically from lifespan."""
    try:
        await _collect_db_pool_metrics()
    except Exception as e:
        log.warning("Failed to collect DB pool metrics: %s", e)
    try:
        await _collect_qdrant_metrics(ml_clients)
    except Exception as e:
        log.warning("Failed to collect Qdrant metrics: [%s] %s", type(e).__name__, e)
    try:
        await _collect_bm25_metrics(ml_clients)
    except Exception as e:
        log.warning("Failed to collect BM25 metrics: [%s] %s", type(e).__name__, e)
    try:
        await _collect_ollama_metrics()
    except Exception as e:
        log.warning("Failed to collect Ollama metrics: [%s] %s", type(e).__name__, e)
