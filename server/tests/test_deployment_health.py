"""Readiness fails closed while the legacy health endpoint remains compatible."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
import httpx
import pytest

from application.health.ports import HealthCheckResult
from application.health.service import HealthService
from application.runtime.enums import HealthStatus
from infrastructure.health.system_health_probe import SystemHealthProbe
from presentation.api.dependencies import create_health_service
from presentation.api.routes.health import router
from tests.fakes import FakeUnitOfWorkFactory


@pytest.fixture
def health_service() -> tuple[HealthService, SimpleNamespace]:
    probe = SimpleNamespace(
        **{
            name: AsyncMock(return_value=HealthCheckResult(status=HealthStatus.OK.value))
            for name in ("check_qdrant", "check_postgres", "check_redis", "check_openrouter")
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
        SimpleNamespace(version="test", uptime_seconds=1, llm_provider="openrouter", ml_provider="tei"),
    )
    return service, probe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_check",
    [None, "check_postgres", "check_qdrant", "check_redis", "check_openrouter", "tei_embed", "tei_rerank"],
)
async def test_readiness_dependency_failure(
    health_service: tuple[HealthService, SimpleNamespace], failed_check: str | None
) -> None:
    service, probe = health_service

    if failed_check in ("tei_embed", "tei_rerank"):
        probe.check_tei.return_value[failed_check] = HealthCheckResult(status=HealthStatus.ERROR.value)
    elif failed_check:
        getattr(probe, failed_check).return_value = HealthCheckResult(status=HealthStatus.ERROR.value)

    app = FastAPI()
    app.include_router(router)

    async def dependency() -> Any:
        return service

    app.dependency_overrides[create_health_service] = dependency
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:  # type: ignore[arg-type]  # Starlette and HTTPX disagree on the ASGI send annotation.
        assert (await client.get("/ready")).status_code == (503 if failed_check else 200)
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/live")).status_code == 200


@pytest.mark.asyncio
async def test_liveness_never_constructs_dependency_service() -> None:
    app = FastAPI()
    app.include_router(router)

    async def unavailable() -> None:
        msg = "Dependencies unavailable"
        raise RuntimeError(msg)

    app.dependency_overrides[create_health_service] = unavailable
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:  # type: ignore[arg-type]  # Starlette and HTTPX disagree on the ASGI send annotation.
        assert (await client.get("/live")).status_code == 200


@pytest.mark.asyncio
async def test_remote_provider_does_not_require_tei(
    health_service: tuple[HealthService, SimpleNamespace],
) -> None:
    service, probe = health_service
    service._settings.ml_provider = "deepinfra"  # type: ignore[misc]  # Deliberate frozen-field write must fail at runtime.
    result = await service.check()
    probe.check_tei.assert_not_awaited()
    assert "tei_embed" not in result.checks


@pytest.mark.asyncio
async def test_tei_probe_rejects_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _request: httpx.Response(503))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
    result = await SystemHealthProbe().check_tei()
    assert all(check.status.startswith(HealthStatus.ERROR.value) for check in result.values())
