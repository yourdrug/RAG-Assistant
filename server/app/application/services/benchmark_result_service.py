"""Compatibility exports for application.benchmark."""

from application.benchmark import (
    BenchmarkResultDetail,
    BenchmarkResultService,
    BenchmarkResultsList,
    BenchmarkResultSummary,
    normalize_metrics,
)


__all__ = [
    "BenchmarkResultDetail",
    "BenchmarkResultService",
    "BenchmarkResultSummary",
    "BenchmarkResultsList",
    "normalize_metrics",
]
