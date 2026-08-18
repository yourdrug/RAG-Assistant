"""Metrics registry port — abstract interface for Prometheus metrics collection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsRegistryPort(Protocol):
    """Abstract interface for collecting Prometheus metrics."""

    def collect_gauge(self, name: str) -> dict[str, float]: ...
    def collect_counter(self, name: str) -> dict[str, float]: ...
    def collect_histogram(self, name: str) -> dict[str, object]: ...
