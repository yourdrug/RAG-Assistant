"""Rate limiting configuration — per-user limits backed by Redis.

Uses fastapi-limiter v0.1.x with a custom identifier that extracts user ID
from the JWT token when available, falling back to IP address.
"""

from __future__ import annotations

import jwt as _jwt
from config import settings
from fastapi import Request
from fastapi_limiter.depends import RateLimiter


async def _user_identifier(request: Request) -> str:
    """Identify rate limit key by user ID (from JWT) or client IP."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            payload = _jwt.decode(
                auth_header[7:],
                settings.jwt_secret_key,
                algorithms=["HS256"],
            )
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except _jwt.InvalidTokenError:
            pass

    # Fallback to IP + path
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0]
    elif request.client:
        ip = request.client.host
    else:
        ip = "127.0.0.1"
    return f"ip:{ip}:{request.scope['path']}"


chat_rate_limit = RateLimiter(times=20, seconds=60, identifier=_user_identifier)
upload_rate_limit = RateLimiter(times=10, seconds=60, identifier=_user_identifier)
ingest_rate_limit = RateLimiter(times=10, seconds=60, identifier=_user_identifier)
