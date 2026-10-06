"""Question imports must not consume the quota for expensive benchmark jobs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.ports.rate_limit import RateLimitDecision, RateLimitPolicyName
from domain.exceptions import ClientException
from domain.value_objects.roles import UserKind, UserRole
from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import create_benchmark_question_service, create_rate_limiter
from presentation.api.exception_handlers import handle_client_exception
from presentation.api.routes.benchmark_admin import router
from presentation.api.schemas import CurrentUser


@pytest.mark.parametrize("valid,allow_write", [(True, True), (False, True), (True, False)])
def test_import_uses_write_quota_even_when_benchmark_quota_is_exhausted(valid, allow_write):
    limiter = SimpleNamespace(
        check=AsyncMock(
            side_effect=lambda policy, principal: RateLimitDecision(
                allowed=policy == RateLimitPolicyName.WRITE and allow_write,
                retry_after_sec=60,
                limit=60,
            )
        )
    )
    service = SimpleNamespace(bulk_create=AsyncMock(return_value=1))
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ClientException, handle_client_exception)
    user = CurrentUser(
        id=7,
        email="admin@example.com",
        role=UserRole.ADMIN.value,
        kind=UserKind.INTERNAL.value,
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[create_rate_limiter] = lambda: limiter
    app.dependency_overrides[create_benchmark_question_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/admin/benchmark/questions/import",
            json={"questions": [{"question": "Q" if valid else "", "is_active": True}]},
        )

    expected_status = (200 if valid else 422) if allow_write else 429
    assert response.status_code == expected_status
    limiter.check.assert_awaited_once_with(
        RateLimitPolicyName.WRITE, f"user:{UserKind.INTERNAL.value}:{user.id}"
    )
    if valid and allow_write:
        assert response.json() == {"imported": 1}
        service.bulk_create.assert_awaited_once()
    else:
        service.bulk_create.assert_not_awaited()
    if not allow_write:
        assert response.json()["errors"]["code"] == "rate_limit_exceeded"
        assert response.headers["Retry-After"] == "60"
