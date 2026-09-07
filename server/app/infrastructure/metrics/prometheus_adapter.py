"""Prometheus metrics adapter — wraps prometheus_client behind MetricsRegistryPort."""

from __future__ import annotations

from prometheus_client import REGISTRY


class PrometheusMetricsRegistry:
    """Adapts prometheus_client REGISTRY behind MetricsRegistryPort."""

    def collect_gauge(self, name: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for metric in REGISTRY.collect():
            if metric.name == name:
                for sample in metric.samples:
                    key = name
                    if sample.labels:
                        key = "_".join(f"{v}" for v in sample.labels.values())
                    result[key] = sample.value
        return result

    def collect_counter(self, name: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for metric in REGISTRY.collect():
            if metric.name == name:
                for sample in metric.samples:
                    if sample.name.endswith("_total") or sample.name == name:
                        key = "_".join(f"{v}" for v in sample.labels.values()) if sample.labels else "total"
                        result[key] = sample.value
        return result

    def collect_histogram(self, name: str) -> dict[str, object]:
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
