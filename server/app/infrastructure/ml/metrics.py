"""Prometheus metrics and collectors for the RAG pipeline and infrastructure.

Defines counters, histograms, and gauges for RAG query quality, document
ingestion throughput, and infrastructure health (Postgres pool, Qdrant
collection size, Ollama GPU usage).  A periodic background collector
refreshes the infrastructure gauges every 30 seconds.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx
from config import settings
from prometheus_client import Counter, Gauge, Histogram

from infrastructure.clients import get_bm25_index, get_qdrant_client
from infrastructure.database.database import database

if TYPE_CHECKING:
    pass

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


async def collect_infra_metrics() -> None:
    """Update infrastructure gauges. Called periodically from lifespan."""
    # Postgres pool
    try:
        engine = database.master_node.async_engine  # type: ignore[union-attr]
        pool = engine.pool
        DB_POOL_IN_USE.set(pool.checked_out())
        DB_POOL_IDLE.set(pool.checked_in())
        DB_POOL_OVERFLOW.set(pool.overflow())
    except Exception:
        pass

    # Qdrant collection size
    try:
        client = get_qdrant_client()
        info = client.get_collection(settings.collection_name)
        QDRANT_POINTS.set(info.points_count or 0)
    except Exception:
        pass

    # BM25 index size
    try:
        bm25 = get_bm25_index()
        if bm25 is not None:
            BM25_INDEX_SIZE.set(len(bm25.hashes))
    except Exception:
        pass

    # Ollama GPU/RAM usage
    try:
        async with httpx.AsyncClient(timeout=3) as http:
            r = await http.get(f"{settings.ollama_base_url}/api/ps")
            if r.status_code == 200:
                data = r.json()
                for proc in data.get("models", []):
                    model_name = proc.get("name", "unknown")
                    vram = proc.get("size_vram", 0)
                    ram = proc.get("size", 0)
                    OLLAMA_GPU_MEMORY_BYTES.labels(model=model_name).set(vram)
                    OLLAMA_RAM_MEMORY_BYTES.labels(model=model_name).set(ram)
    except Exception:
        pass


async def _periodic_infra_collector(interval: float = 30.0) -> None:
    """Background task that updates infrastructure gauges periodically."""
    while True:
        await asyncio.sleep(interval)
        await collect_infra_metrics()
