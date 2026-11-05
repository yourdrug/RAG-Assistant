"""Job errors remain available by ID and are restricted to administrators."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from application.jobs.service import JobService
from domain.identity.roles import UserKind, UserRole
from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus
from domain.shared.exceptions import AppClientError, EntityNotFoundError
from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import create_job_service
from presentation.api.exception_handlers import handle_client_exception
from presentation.api.routes.admin_jobs import router
from presentation.api.schemas import CurrentUser
from tests.fakes import FakeUnitOfWorkFactory


@pytest.mark.parametrize("role", list(UserRole))
def test_job_detail_exposes_full_error_only_to_admin(role: UserRole) -> None:
    error = "Circuit breaker 'llm_auxiliary' not initialised. Call init_breakers() at startup."
    job = BackgroundJob(id=17, status=BackgroundJobStatus.FAILED, error_message=error)
    service = SimpleNamespace(get=AsyncMock(return_value=job))
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(AppClientError, handle_client_exception)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        id=7, email="user@example.com", role=role.value, kind=UserKind.INTERNAL.value, is_active=True
    )
    app.dependency_overrides[create_job_service] = lambda: service
    with TestClient(app) as client:
        response = client.get("/admin/jobs/17")

    if role == UserRole.ADMIN:
        assert response.status_code == 200
        assert response.json()["error_message"] == error
        assert response.json()["status"] == BackgroundJobStatus.FAILED.value
        service.get.assert_awaited_once_with(17)
    else:
        assert response.status_code == 403
        service.get.assert_not_called()


@pytest.mark.asyncio
async def test_missing_job_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = FakeUnitOfWorkFactory()
    monkeypatch.setattr(factory._uow.fake_background_jobs, "get_by_id", AsyncMock(return_value=None))
    with pytest.raises(EntityNotFoundError):
        await JobService(factory).get(17)
