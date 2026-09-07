"""Prometheus metrics — counters, histograms, gauges for RAG pipeline and infrastructure."""

from infrastructure.metrics.metrics import (
    INGEST_FILES_TOTAL,
    RAG_QUERIES_TOTAL,
    collect_infra_metrics,
)
from infrastructure.metrics.metrics_adapter import PrometheusMetricsCollector
from infrastructure.metrics.prometheus_adapter import PrometheusMetricsRegistry

__all__ = [
    "INGEST_FILES_TOTAL",
    "RAG_QUERIES_TOTAL",
    "PrometheusMetricsCollector",
    "PrometheusMetricsRegistry",
    "collect_infra_metrics",
]
