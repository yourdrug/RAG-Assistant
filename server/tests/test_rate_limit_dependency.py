"""Tests for the presentation-layer rate limit dependency and 429 contract."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Annotated, Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from application.ports.rate_limit import RateLimitPolicyName
from domain.identity.roles import UserKind, UserRole
from domain.shared.exceptions import AppClientError, AuthenticationError
from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import create_rate_limiter
from presentation.api.exception_handlers import handle_client_exception
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import CurrentUser
from tests.fakes import FakeRateLimiterPort


def _current_user() -> CurrentUser:
    return CurrentUser(
        id=7,
        email="user@example.com",
        role=UserRole.USER.value,
        kind=UserKind.INTERNAL.value,
        is_active=True,
    )


def _unauthorized() -> CurrentUser:
    msg = "Not authenticated"
    raise AuthenticationError(msg)


def _client(limiter: Any, *, authenticate: bool = True) -> TestClient:
    app = FastAPI()
    app.add_exception_handler(AppClientError, handle_client_exception)

    @app.post("/limited", dependencies=[Depends(rate_limit(RateLimitPolicyName.SEARCH))])
    async def limited(current_user: Annotated[CurrentUser, Depends(get_current_user)]) -> dict[str, Any]:
        return {"ok": True}

    @app.post("/login", dependencies=[Depends(rate_limit(RateLimitPolicyName.LOGIN))])
    async def login() -> dict[str, Any]:
        return {"ok": True}

    if authenticate:
        app.dependency_overrides[get_current_user] = _current_user
    else:
        app.dependency_overrides[get_current_user] = _unauthorized

    app.dependency_overrides[create_rate_limiter] = lambda: limiter
    return TestClient(app, raise_server_exceptions=False)


class TestUserScopedPolicy:
    def test_allowed_request_passes_and_keys_by_user(self) -> None:
        limiter = FakeRateLimiterPort()
        response = _client(limiter).post("/limited")

        assert response.status_code == 200
        assert limiter.calls == [("search", "user:internal:7")]

    def test_exceeded_returns_429_envelope_with_headers(self) -> None:
        limiter = FakeRateLimiterPort(allow=0)
        response = _client(limiter).post("/limited")

        assert response.status_code == 429
        body = response.json()
        assert body["errors"]["code"] == "rate_limit_exceeded"
        assert "message" in body
        assert response.headers["Retry-After"] == "60"
        assert response.headers["X-RateLimit-Limit"] == "1"

    def test_requires_authentication_before_limit_check(self) -> None:
        limiter = FakeRateLimiterPort()
        response = _client(limiter, authenticate=False).post("/limited")

        assert response.status_code == 401
        assert limiter.calls == []

    def test_disabled_limiter_is_noop(self) -> None:
        response = _client(None).post("/limited")

        assert response.status_code == 200


class TestIpScopedPolicy:
    def test_login_keys_by_client_ip_without_auth(self) -> None:
        limiter = FakeRateLimiterPort()
        response = _client(limiter, authenticate=False).post("/login")

        assert response.status_code == 200
        assert limiter.calls == [("login", "ip:testclient")]

    def test_login_exceeded_returns_429(self) -> None:
        limiter = FakeRateLimiterPort(allow=0)
        response = _client(limiter, authenticate=False).post("/login")

        assert response.status_code == 429
        assert response.json()["errors"]["code"] == "rate_limit_exceeded"
