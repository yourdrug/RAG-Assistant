"""Prometheus auto-instrumentation middleware for FastAPI.

Wraps ``prometheus_fastapi_instrumentator`` to expose a ``/metrics``
endpoint and automatically record request latency / status-code histograms.
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def add_metrics_middleware(app: FastAPI) -> None:
    """Add Prometheus auto-instrumentation and expose /metrics endpoint."""
    Instrumentator(
        excluded_handlers=["/metrics", "/health"],
        should_group_status_codes=False,
        should_group_untemplated=True,
    ).instrument(app).expose(app, endpoint="/metrics")
