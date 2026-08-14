"""HTTP request metrics middleware — increments prometheus_client counters."""

from __future__ import annotations

from infrastructure.ml.metrics import HTTP_REQUESTS_TOTAL
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        handler = request.url.path
        method = request.method
        status = str(response.status_code)
        HTTP_REQUESTS_TOTAL.labels(handler=handler, method=method, status=status).inc()
        return response
