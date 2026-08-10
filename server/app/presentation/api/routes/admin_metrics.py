"""Admin metrics endpoint — Prometheus metrics as JSON."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from prometheus_client import REGISTRY

from presentation.api.auth_dependencies import require_admin
from presentation.api.schemas import MetricsResponse

router = APIRouter(tags=["admin-metrics"])


def _collect_gauge(name: str) -> dict[str, float]:
    """Collect all samples for a gauge metric into {label_key: value}."""
    result: dict[str, float] = {}
    for metric in REGISTRY.collect():
        if metric.name == name:
            for sample in metric.samples:
                key = name
                if sample.labels:
                    key = "_".join(f"{v}" for v in sample.labels.values())
                result[key] = sample.value
    return result


def _collect_counter(name: str) -> dict[str, float]:
    """Collect counter totals by label combination."""
    result: dict[str, float] = {}
    for metric in REGISTRY.collect():
        if metric.name == name:
            for sample in metric.samples:
                if sample.name.endswith("_total") or sample.name == name:
                    key = "_".join(f"{v}" for v in sample.labels.values()) if sample.labels else "total"
                    result[key] = sample.value
    return result


def _collect_histogram(name: str) -> dict[str, object]:
    """Collect histogram summary (count, sum, buckets)."""
    result: dict[str, object] = {}
    for metric in REGISTRY.collect():
        if metric.name == name:
            for sample in metric.samples:
                if sample.name == f"{name}_count":
                    result["count"] = sample.value
                elif sample.name == f"{name}_sum":
                    result["sum"] = sample.value
                elif sample.name.endswith("_bucket"):
                    bucket = sample.name.split("_")[-2] if "_bucket" in sample.name else "unknown"
                    result[f"bucket_{bucket}"] = sample.value
    return result


@router.get("/admin/metrics", response_model=MetricsResponse)
async def get_metrics(admin: dict = Depends(require_admin)):
    # Database pool
    db_pool = {}
    for name in ["db_pool_connections_in_use", "db_pool_connections_idle", "db_pool_overflow"]:
        vals = _collect_gauge(name)
        db_pool[name.replace("db_pool_", "")] = sum(vals.values()) if vals else 0.0

    # Qdrant
    qdrant_vals = _collect_gauge("qdrant_collection_points")
    qdrant = {"points": sum(qdrant_vals.values()) if qdrant_vals else 0.0}

    # BM25
    bm25_vals = _collect_gauge("bm25_index_size")
    bm25 = {"index_size": sum(bm25_vals.values()) if bm25_vals else 0.0}

    # Ollama GPU/RAM
    ollama = []
    gpu_vals = _collect_gauge("ollama_gpu_memory_bytes")
    ram_vals = _collect_gauge("ollama_ram_memory_bytes")
    all_models = set(list(gpu_vals.keys()) + list(ram_vals.keys()))
    for model in all_models:
        ollama.append(
            {
                "model": model,
                "gpu_bytes": gpu_vals.get(model, 0.0),
                "ram_bytes": ram_vals.get(model, 0.0),
            }
        )

    # RAG metrics
    rag_queries = _collect_counter("rag_queries_total")
    rag_latency = _collect_histogram("rag_stage_duration_seconds")
    rag_answer_len = _collect_histogram("rag_answer_length_chars")
    rag_chunks = _collect_histogram("rag_retrieved_chunks")
    rag_not_found = _collect_counter("rag_not_found_total")
    rag = {
        "queries_total": sum(rag_queries.values()) if rag_queries else 0.0,
        "not_found_total": sum(rag_not_found.values()) if rag_not_found else 0.0,
        "stage_latency": rag_latency,
        "answer_length": rag_answer_len,
        "retrieved_chunks": rag_chunks,
    }

    # Ingestion metrics (total + by status)
    ingest_docs = _collect_counter("ingest_documents_total")
    ingest_chunks = _collect_counter("ingest_chunks_total")
    ingest_files = _collect_counter("ingest_files_total")
    ingest_duration = _collect_histogram("ingest_document_duration_seconds")
    ingestion = {
        "documents_total": sum(ingest_docs.values()) if ingest_docs else 0.0,
        "chunks_total": sum(ingest_chunks.values()) if ingest_chunks else 0.0,
        "files_total": sum(ingest_files.values()) if ingest_files else 0.0,
        "duration": ingest_duration,
    }
    # Per-status breakdown from labels
    ingest_by_status: dict[str, float] = {}
    for metric in REGISTRY.collect():
        if metric.name == "ingest_documents_total":
            for sample in metric.samples:
                if sample.name.endswith("_total") or sample.name == "ingest_documents_total":
                    status = sample.labels.get("status", "unknown") if sample.labels else "unknown"
                    ingest_by_status[status] = sample.value
    ingestion["by_status"] = ingest_by_status

    # HTTP requests
    http_requests_raw = _collect_counter("http_requests_total")
    # Aggregate by handler
    by_handler: dict[str, float] = {}
    total_requests = 0.0
    for key, val in http_requests_raw.items():
        total_requests += val
        # key is like "handler_method_status"
        parts = key.split("_")
        if len(parts) >= 2:
            handler = parts[0]  # first part is the handler path
        else:
            handler = key
        by_handler[handler] = by_handler.get(handler, 0) + val
    http_requests = {
        "total": total_requests,
        "by_endpoint": http_requests_raw,
    }

    return MetricsResponse(
        db_pool=db_pool,
        qdrant=qdrant,
        bm25=bm25,
        ollama=ollama,
        rag=rag,
        ingestion=ingestion,
        http_requests=http_requests,
    )
