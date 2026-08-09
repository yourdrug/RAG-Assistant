"""Rate limiting middleware — per-user/IP token-bucket limiter for expensive endpoints."""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse


class _TokenBucket:
    """Thread-safe token bucket for a single key."""

    __slots__ = ("capacity", "tokens", "refill_rate", "last_refill", "_lock")

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()
        self._lock = Lock()

    def consume(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-user/IP rate limiter applied to specific routes.

    Default: 20 requests/minute per user for chat endpoints, 10/minute for upload.
    """

    def __init__(
        self,
        app,
        chat_limit: int = 20,
        upload_limit: int = 10,
        window_seconds: float = 60.0,
    ) -> None:
        super().__init__(app)
        self._buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(capacity=chat_limit, refill_rate=chat_limit / window_seconds)
        )
        self._upload_buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(capacity=upload_limit, refill_rate=upload_limit / window_seconds)
        )
        self._lock = Lock()
        self._chat_paths = ("/chat", "/chat/sync")
        self._upload_paths = ("/documents", "/ingest")

    def _get_key(self, request: Request) -> str:
        user = getattr(request.state, "user", None)
        if user and "id" in user:
            return f"user:{user['id']}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        is_chat = any(path.rstrip("/") == p for p in self._chat_paths)
        is_upload = path.rstrip("/") == "/documents" and request.method == "POST"

        if not (is_chat or is_upload):
            return await call_next(request)

        key = self._get_key(request)

        if is_upload:
            with self._lock:
                bucket = self._upload_buckets[key]
            allowed = bucket.consume()
        else:
            with self._lock:
                bucket = self._buckets[key]
            allowed = bucket.consume()

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={"Retry-After": "60"},
            )

        return await call_next(request)
