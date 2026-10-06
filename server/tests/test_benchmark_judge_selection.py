"""Judge selection survives the API, persistence mapping and worker boundary."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from application.ports.rate_limit import RateLimitDecision
from application.services.benchmark_services import BenchmarkSweepService
from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.exceptions import ClientException
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.sweep_evaluation_mode import SweepEvaluationMode
from fakes import FakeUnitOfWorkFactory
from infrastructure.worker import sweep as worker
from presentation.api.auth_dependencies import get_current_user
from presentation.api.exception_handlers import handle_client_exception
from presentation.api.dependencies import (
    create_benchmark_sweep_service,
    create_job_enqueuer,
    create_job_service,
    create_rate_limiter,
)
from presentation.api.routes.benchmark_admin import router
from presentation.api.schemas import CurrentUser
from presentation.api.schemas.benchmark import SweepCreateRequest


@pytest.mark.parametrize("model", ["", "  ", "a" * 256])
def test_blank_or_oversized_judge_model_is_rejected(model):
    with pytest.raises(ValidationError):
        SweepCreateRequest(search_space={}, judge_model=model)


@pytest.mark.parametrize("role", [UserRole.ADMIN, UserRole.USER])
@pytest.mark.parametrize("mode", list(SweepEvaluationMode))
def test_api_preserves_selected_judge_and_requires_admin(role, mode):
    factory = FakeUnitOfWorkFactory()
    service = BenchmarkSweepService(factory)
    jobs = SimpleNamespace(create_job=AsyncMock(return_value=2))
    queue = SimpleNamespace(enqueue_sweep=AsyncMock())
    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(ClientException, handle_client_exception)
    user = CurrentUser(
        id=7, email="user@example.com", role=role.value, kind=UserKind.INTERNAL.value, is_active=True
    )
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[create_benchmark_sweep_service] = lambda: service
    app.dependency_overrides[create_job_service] = lambda: jobs
    app.dependency_overrides[create_job_enqueuer] = lambda: queue
    app.dependency_overrides[create_rate_limiter] = lambda: SimpleNamespace(
        check=AsyncMock(return_value=RateLimitDecision(allowed=True, retry_after_sec=0, limit=10))
    )
    with TestClient(app) as client:
        response = client.post(
            "/admin/benchmark/sweep",
            json={
                "search_space": {"top_k": {"values": [2]}},
                "judge_model": " chosen/judge ",
                "evaluation_mode": mode.value,
            },
        )
    if role == UserRole.ADMIN:
        assert response.status_code == 200, response.text
        assert response.json()["judge_model"] == "chosen/judge"
        assert response.json()["evaluation_mode"] == mode
        queue.enqueue_sweep.assert_awaited_once_with(sweep_id=1, job_id=2)
    else:
        assert response.status_code == 403
        jobs.create_job.assert_not_called()
        queue.enqueue_sweep.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selected,configured,expected",
    [
        ("chosen/judge", "configured/judge", "chosen/judge"),
        (None, "configured/judge", "configured/judge"),
        (None, "", "main-model"),
    ],
)
async def test_worker_uses_saved_judge_instead_of_main_model(monkeypatch, selected, configured, expected):
    from infrastructure.worker import tasks

    saved = BenchmarkSweep(id=1, judge_model=selected)
    factory = FakeUnitOfWorkFactory()
    factory._uow.benchmark_sweeps.get_by_id = AsyncMock(return_value=saved)
    engine = SimpleNamespace(run_sweep=AsyncMock(return_value=[]))
    monkeypatch.setattr(worker, "create_sweep_engine", MagicMock(return_value=engine))
    monkeypatch.setattr(worker, "_publish_sweep_event", AsyncMock())
    monkeypatch.setattr(worker.settings, "llm_model", "main-model")
    monkeypatch.setattr(worker.settings, "benchmark_judge_model", configured)

    async def run_action(factory, job_id, action, **kwargs):
        await action()

    monkeypatch.setattr(tasks, "_run_tracked_job", run_action)
    container = SimpleNamespace(infrastructure=SimpleNamespace(db=SimpleNamespace(uow_factory=factory)))
    await worker.run_sweep_task({"container": container}, sweep_id=1, job_id=2)
    assert engine.run_sweep.call_args.kwargs["judge_model"] == expected
