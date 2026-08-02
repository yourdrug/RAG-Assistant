"""Metrics middleware — Prometheus Instrumentator setup."""

from __future__ import annotations

from fastapi import FastAPI


def add_metrics_middleware(app: FastAPI) -> None:
    """Add Prometheus auto-instrumentation and expose /metrics endpoint."""
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        excluded_handlers=["/metrics", "/health"],
        should_group_status_codes=False,
        should_group_untemplated=True,
    ).instrument(app).expose(app, endpoint="/metrics")
