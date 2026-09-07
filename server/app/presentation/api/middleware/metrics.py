"""HTTP request metrics middleware — increments prometheus_client counters.

M-4: the handler label uses the ROUTE TEMPLATE (e.g. ``/documents/{id}``),
never the raw request path — raw paths with object ids grow the Prometheus
registry without bound (label-value cardinality bomb).
"""

from __future__ import annotations

from infrastructure.metrics.metrics import HTTP_REQUESTS_TOTAL
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_UNMATCHED = "unmatched"


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        route = request.scope.get("route")
        # path_format is the templated path ("/documents/{document_id}");
        # requests that matched no route collapse into a single bounded label.
        handler = getattr(route, "path_format", None) or getattr(route, "path", None) or _UNMATCHED
        method = request.method
        status = str(response.status_code)
        HTTP_REQUESTS_TOTAL.labels(handler=handler, method=method, status=status).inc()
        return response
