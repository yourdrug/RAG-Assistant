"""Application service for Prometheus metrics aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field

from application.ports.metrics_registry import MetricsRegistryPort


@dataclass(frozen=True)
class MetricsResult:
    db_pool: dict[str, float] = field(default_factory=dict)
    qdrant: dict[str, float] = field(default_factory=dict)
    bm25: dict[str, float] = field(default_factory=dict)
    ollama: list[dict[str, object]] = field(default_factory=list)
    rag: dict[str, object] = field(default_factory=dict)
    ingestion: dict[str, object] = field(default_factory=dict)
    http_requests: dict[str, object] = field(default_factory=dict)


class MetricsService:
    def __init__(self, registry: MetricsRegistryPort) -> None:
        self._registry = registry

    def collect(self) -> MetricsResult:
        reg = self._registry

        # Database pool
        db_pool = {}
        for name in ["db_pool_connections_in_use", "db_pool_connections_idle", "db_pool_overflow"]:
            vals = reg.collect_gauge(name)
            db_pool[name.replace("db_pool_", "")] = sum(vals.values()) if vals else 0.0

        # Qdrant
        qdrant_vals = reg.collect_gauge("qdrant_collection_points")
        qdrant = {"points": sum(qdrant_vals.values()) if qdrant_vals else 0.0}

        # BM25
        bm25_vals = reg.collect_gauge("bm25_index_size")
        bm25 = {"index_size": sum(bm25_vals.values()) if bm25_vals else 0.0}

        # Ollama GPU/RAM
        ollama = []
        gpu_vals = reg.collect_gauge("ollama_gpu_memory_bytes")
        for model, gpu_bytes in gpu_vals.items():
            ollama.append(
                {
                    "model": model,
                    "gpu_bytes": gpu_bytes,
                }
            )

        # RAG metrics
        rag_queries = reg.collect_counter("rag_queries")
        rag_latency = reg.collect_histogram("rag_stage_duration_seconds")
        rag_answer_len = reg.collect_histogram("rag_answer_length_chars")
        rag_chunks = reg.collect_histogram("rag_retrieved_chunks")
        rag_not_found = reg.collect_counter("rag_not_found")
        rag: dict[str, object] = {
            "queries_total": sum(rag_queries.values()) if rag_queries else 0.0,
            "not_found_total": sum(rag_not_found.values()) if rag_not_found else 0.0,
            "stage_latency": rag_latency,
            "answer_length": rag_answer_len,
            "retrieved_chunks": rag_chunks,
        }

        # Ingestion metrics
        ingest_docs = reg.collect_counter("ingest_documents")
        ingest_chunks = reg.collect_counter("ingest_chunks")
        ingest_files = reg.collect_counter("ingest_files")
        ingest_duration = reg.collect_histogram("ingest_document_duration_seconds")
        ingest_pdf_pages = reg.collect_counter("ingest_pdf_pages")
        ingest_pdf_bad_ratio = reg.collect_histogram("ingest_pdf_bad_ratio")

        ingestion: dict[str, object] = {
            "documents_total": sum(ingest_docs.values()) if ingest_docs else 0.0,
            "chunks_total": sum(ingest_chunks.values()) if ingest_chunks else 0.0,
            "files_total": sum(ingest_files.values()) if ingest_files else 0.0,
            "duration": ingest_duration,
            "pdf_pages": ingest_pdf_pages,
            "pdf_bad_ratio": ingest_pdf_bad_ratio,
        }

        # Per-status breakdown
        ingest_by_status: dict[str, float] = {}
        for key, val in ingest_docs.items():
            parts = key.split("_")
            if len(parts) >= 2:
                status = parts[-1]
            else:
                status = key
            ingest_by_status[status] = val
        ingestion["by_status"] = ingest_by_status

        # HTTP requests
        http_requests_raw = reg.collect_counter("http_requests")
        total_requests = sum(http_requests_raw.values())
        http_requests: dict[str, object] = {
            "total": total_requests,
            "by_endpoint": http_requests_raw,
        }

        return MetricsResult(
            db_pool=db_pool,
            qdrant=qdrant,
            bm25=bm25,
            ollama=ollama,
            rag=rag,
            ingestion=ingestion,
            http_requests=http_requests,
        )
