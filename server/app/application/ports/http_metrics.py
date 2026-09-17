"""Protocol for HTTP request metrics counter."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HttpMetricsPort(Protocol):
    """Increments HTTP request counters (used by MetricsMiddleware)."""

    def inc_requests(self, *, handler: str, method: str, status: str) -> None: ...
