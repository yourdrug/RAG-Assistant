"""Rate limiting configuration — per-user limits backed by Redis.

Uses fastapi-limiter v0.1.x with a custom identifier that extracts user ID
from the JWT token when available, falling back to the trusted client IP.

Client IP is taken from the socket address only. uvicorn rewrites
``request.client`` from X-Forwarded-For solely when the direct peer is a
trusted proxy (``FORWARDED_ALLOW_IPS``), so a client cannot spoof the
rate-limit key by sending its own X-Forwarded-For header.
"""

from __future__ import annotations

import jwt as _jwt
from config import settings
from fastapi import Request
from fastapi_limiter.depends import RateLimiter


async def _client_ip(request: Request) -> str:
    if request.client:
        return request.client.host
    return "unknown"


async def _client_ip_identifier(request: Request) -> str:
    return f"ip:{await _client_ip(request)}:{request.scope['path']}"


async def _user_identifier(request: Request) -> str:
    """Identify rate limit key by user ID (from JWT) or trusted client IP."""
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

    return await _client_ip_identifier(request)


async def _login_email_identifier(request: Request) -> str:
    """Per-account login limit — protects a single mailbox from brute force.

    The body is read (and cached by Starlette) so FastAPI's own validation
    of the LoginRequest body still works afterwards. A malformed body falls
    back to the shared "unknown" bucket; the endpoint rejects it with 422.
    """
    try:
        body = await request.json()
    except ValueError:
        body = None
    email = str(body.get("email", "")).strip().lower() if isinstance(body, dict) else ""
    return f"email:{email or 'unknown'}:{request.scope['path']}"


chat_rate_limit = RateLimiter(times=20, seconds=60, identifier=_user_identifier)
upload_rate_limit = RateLimiter(times=10, seconds=60, identifier=_user_identifier)
ingest_rate_limit = RateLimiter(times=10, seconds=60, identifier=_user_identifier)

# POST /auth/login has no JWT yet — limit per IP and per account (brute force).
login_rate_limit = RateLimiter(times=10, seconds=60, identifier=_client_ip_identifier)
login_email_rate_limit = RateLimiter(times=10, seconds=300, identifier=_login_email_identifier)
