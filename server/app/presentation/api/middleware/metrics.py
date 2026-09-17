"""HTTP request metrics middleware — increments prometheus_client counters.

M-4: the handler label uses the ROUTE TEMPLATE (e.g. ``/documents/{id}``),
never the raw request path — raw paths with object ids grow the Prometheus
registry without bound (label-value cardinality bomb).
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_UNMATCHED = "unmatched"


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        route = request.scope.get("route")
        handler = getattr(route, "path_format", None) or getattr(route, "path", None) or _UNMATCHED
        method = request.method
        status = str(response.status_code)
        container = getattr(request.app.state, "container", None)
        http_metrics = getattr(getattr(container, "infrastructure", None), "http_metrics", None)
        if http_metrics is not None:
            http_metrics.inc_requests(handler=handler, method=method, status=status)
        return response
