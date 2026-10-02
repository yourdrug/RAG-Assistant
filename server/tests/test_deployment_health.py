"""Readiness fails closed while the legacy health endpoint remains compatible."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from application.ports.health import HealthCheckResult
from application.services.health_service import HealthService
from domain.value_objects.health_status import HealthStatus
from fakes import FakeUnitOfWorkFactory
from presentation.api.dependencies import create_health_service
from presentation.api.routes.health import router


@pytest.fixture
def health_service():
    probe = SimpleNamespace(
        **{
            name: AsyncMock(return_value=HealthCheckResult(status=HealthStatus.OK.value))
            for name in ("check_qdrant", "check_postgres", "check_redis", "check_ollama", "check_openrouter")
        },
        check_tei=AsyncMock(
            return_value={
                "tei_embed": HealthCheckResult(status=HealthStatus.OK.value),
                "tei_rerank": HealthCheckResult(status=HealthStatus.OK.value),
            }
        ),
    )
    service = HealthService(
        FakeUnitOfWorkFactory(),
        probe,
        SimpleNamespace(is_connected=True),
        SimpleNamespace(version="test", uptime_seconds=1, llm_provider="ollama", ml_provider="tei"),
    )
    return service, probe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_check",
    [None, "check_postgres", "check_qdrant", "check_redis", "check_ollama", "tei_embed", "tei_rerank"],
)
async def test_readiness_dependency_failure(health_service, failed_check):
    service, probe = health_service
    if failed_check in ("tei_embed", "tei_rerank"):
        probe.check_tei.return_value[failed_check] = HealthCheckResult(status=HealthStatus.ERROR.value)
    elif failed_check:
        getattr(probe, failed_check).return_value = HealthCheckResult(status=HealthStatus.ERROR.value)
    app = FastAPI()
    app.include_router(router)

    async def dependency():
        return service

    app.dependency_overrides[create_health_service] = dependency
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/ready")).status_code == (503 if failed_check else 200)
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/live")).status_code == 200


@pytest.mark.asyncio
async def test_liveness_never_constructs_dependency_service():
    app = FastAPI()
    app.include_router(router)

    async def unavailable():
        raise RuntimeError("Dependencies unavailable")

    app.dependency_overrides[create_health_service] = unavailable
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/live")).status_code == 200


@pytest.mark.asyncio
async def test_remote_provider_does_not_require_tei(health_service):
    service, probe = health_service
    service._settings.ml_provider = "deepinfra"
    result = await service.check()
    probe.check_tei.assert_not_awaited()
    assert "tei_embed" not in result.checks


@pytest.mark.asyncio
async def test_tei_probe_rejects_http_failure(monkeypatch):
    from infrastructure.health.system_health_probe import SystemHealthProbe

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
    result = await SystemHealthProbe().check_tei()
    assert all(check.status.startswith(HealthStatus.ERROR.value) for check in result.values())
