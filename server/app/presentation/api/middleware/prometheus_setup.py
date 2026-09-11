"""Prometheus auto-instrumentation middleware for FastAPI.

Wraps ``prometheus_fastapi_instrumentator`` to expose a ``/metrics``
endpoint and automatically record request latency / status-code histograms.

M-4 hardening: ``/metrics`` requires authentication (admin) — it discloses
infrastructure details (stage, hostnames in job labels, request rates) and
must not be reachable by unauthenticated clients.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from presentation.api.auth_dependencies import require_admin


def add_metrics_middleware(app: FastAPI) -> None:
    """Add Prometheus auto-instrumentation and expose an authenticated /metrics endpoint."""
    Instrumentator(
        excluded_handlers=["/metrics", "/health"],
        should_group_status_codes=False,
        should_group_untemplated=True,
    ).instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
        dependencies=[Depends(require_admin)],
    )
