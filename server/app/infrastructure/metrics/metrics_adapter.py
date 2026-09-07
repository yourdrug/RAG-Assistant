"""Metrics adapter — wraps Prometheus metrics behind MetricsCollectorPort."""

from __future__ import annotations

from infrastructure.metrics.metrics import (
    INGEST_CHUNKS_TOTAL,
    INGEST_DOCUMENT_DURATION,
    INGEST_DOCUMENTS_TOTAL,
    INGEST_DOMAIN_AMBIGUOUS_TOTAL,
    INGEST_DOMAIN_CLASSIFICATION_CONFIDENCE,
    INGEST_PDF_BAD_RATIO,
    INGEST_PDF_PAGES_TOTAL,
)


class PrometheusMetricsCollector:
    """Adapts Prometheus metric objects behind the MetricsCollectorPort interface."""

    def inc_chunks(self, count: int) -> None:
        INGEST_CHUNKS_TOTAL.inc(count)

    def inc_documents(self, status: str) -> None:
        INGEST_DOCUMENTS_TOTAL.labels(status=status).inc()

    def observe_duration(self, status: str, seconds: float) -> None:
        INGEST_DOCUMENT_DURATION.labels(status=status).observe(seconds)

    def observe_pdf_pages(self, quality: str, count: int) -> None:
        INGEST_PDF_PAGES_TOTAL.labels(quality=quality).inc(count)

    def observe_pdf_bad_ratio(self, ratio: float) -> None:
        INGEST_PDF_BAD_RATIO.observe(ratio)

    def observe_domain_classification(self, domain: str, level: str, confidence: float) -> None:
        INGEST_DOMAIN_CLASSIFICATION_CONFIDENCE.labels(domain=domain, level=level).observe(confidence)

    def inc_domain_ambiguous(self, candidates: str) -> None:
        INGEST_DOMAIN_AMBIGUOUS_TOTAL.labels(candidates=candidates).inc()
