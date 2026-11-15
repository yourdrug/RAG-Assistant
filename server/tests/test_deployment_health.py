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
from config import settings
from infrastructure.health.system_health_probe import SystemHealthProbe
from presentation.api.dependencies import create_health_service
from presentation.api.routes.health import router
from tests.fakes import FakeUnitOfWorkFactory


@pytest.fixture
def health_service() -> tuple[HealthService, SimpleNamespace]:
    probe = SimpleNamespace(
        **{
            name: AsyncMock(return_value=HealthCheckResult(status=HealthStatus.OK.value))
            for name in ("check_qdrant", "check_postgres", "check_redis", "check_llm")
        },
        check_ml=AsyncMock(
            return_value={
                "embeddings": HealthCheckResult(status=HealthStatus.OK.value),
                "reranker": HealthCheckResult(status=HealthStatus.OK.value),
            }
        ),
    )
    service = HealthService(
        FakeUnitOfWorkFactory(),
        probe,
        SimpleNamespace(is_connected=True),
        SimpleNamespace(version="test", uptime_seconds=1),
    )
    return service, probe


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_check",
    [None, "check_postgres", "check_qdrant", "check_redis", "check_llm", "embeddings", "reranker"],
)
async def test_readiness_dependency_failure(
    health_service: tuple[HealthService, SimpleNamespace], failed_check: str | None
) -> None:
    service, probe = health_service

    if failed_check in ("embeddings", "reranker"):
        probe.check_ml.return_value[failed_check] = HealthCheckResult(status=HealthStatus.ERROR.value)
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
async def test_cloud_ml_endpoints_are_not_health_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx.Request] = []
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503)

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
    monkeypatch.setattr(settings, "embedding_base_url", "https://api.deepinfra.com/v1/openai")
    monkeypatch.setattr(settings, "rerank_base_url", "https://api.deepinfra.com/v1")
    assert await SystemHealthProbe().check_ml() == {}
    assert requests == []


@pytest.mark.asyncio
async def test_self_hosted_ml_probe_rejects_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _request: httpx.Response(503))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
    monkeypatch.setattr(settings, "embedding_base_url", "http://tei-embed:8080/v1")
    monkeypatch.setattr(settings, "rerank_base_url", "http://tei-rerank:8080")
    result = await SystemHealthProbe().check_ml()
    assert set(result) == {"embeddings", "reranker"}
    assert all(check.status.startswith(HealthStatus.ERROR.value) for check in result.values())
