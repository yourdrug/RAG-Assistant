"""Prometheus auto-instrumentation middleware for FastAPI.

Wraps ``prometheus_fastapi_instrumentator`` to expose a ``/metrics``
endpoint and automatically record request latency / status-code histograms.

M-4 hardening: ``/metrics`` requires authentication (admin) — it discloses
infrastructure details (stage, hostnames in job labels, request rates) and
must not be reachable by unauthenticated clients.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator

from config import AuthSection, request_section
from domain.shared.exceptions import AuthenticationError
from presentation.api.auth_dependencies import get_current_user, parse_auth_header_value, require_admin
from presentation.api.dependencies import create_api_key_provider, create_api_key_service, create_auth_service


async def require_metrics_access(request: Request) -> None:
    authorization = request.headers.get("Authorization")

    if authorization is None:
        raise AuthenticationError

    parsed = parse_auth_header_value(authorization)
    token = request_section(AuthSection).metrics_scrape_token.get_secret_value()

    if (
        token
        and parsed
        and parsed[0].lower() == "bearer"
        and hmac.compare_digest(token.encode(), parsed[1].encode())
    ):
        return

    # Ordinary administrators retain access; the scrape credential is accepted
    # only by this dependency, never by application authentication.
    user = await get_current_user(
        authorization=authorization,
        auth_service=create_auth_service(request),
        api_key_provider=create_api_key_provider(request),
        api_key_service=create_api_key_service(request),
    )
    require_admin(user)


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
        dependencies=[Depends(require_metrics_access)],
    )
