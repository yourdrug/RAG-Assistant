"""Health and admin adapters share checks and preserve failure responses."""

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from application.health.ports import HealthCheckResult
from infrastructure.health import system_health_probe as probes


HttpFixture = tuple[probes.SystemHealthProbe, dict[str, httpx.Response | Exception], list[httpx.Request]]


@pytest.fixture
def http_probe(monkeypatch: pytest.MonkeyPatch) -> HttpFixture:
    requests: list[httpx.Request] = []
    responses: dict[str, httpx.Response | Exception] = {}
    original_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = responses[request.url.path]

        if isinstance(response, Exception):
            raise response

        return response

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
    monkeypatch.setattr(
        probes,
        "settings",
        SimpleNamespace(
            ollama_base_url="http://ollama",
            openrouter_base_url="http://openrouter",
            openrouter_api_key="test-key",
            openrouter_model="test-model",
            tei_embed_url="http://embed/",
            tei_rerank_url="http://rerank/",
        ),
    )
    return probes.SystemHealthProbe(), responses, requests


@pytest.mark.asyncio
async def test_ollama_models_shared_with_admin(http_probe: HttpFixture) -> None:
    health, responses, requests = http_probe
    responses["/api/tags"] = httpx.Response(200, json={"models": [{"name": "one"}, {"name": "two"}]})
    result = await health.check_ollama()
    assert result.status == "ok"
    assert result.models == ["one", "two"]
    assert result.latency_ms is not None
    assert result.latency_ms >= 0
    assert await probes.OllamaProbe(health).get_models() == result.models
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, content=b"invalid json"),
        httpx.Response(200, json={"models": [{}]}),
        httpx.ConnectError("unavailable"),
    ],
)
async def test_ollama_failure_degrades_to_empty_admin_models(
    http_probe: HttpFixture, response: httpx.Response | Exception
) -> None:
    health, responses, _ = http_probe
    responses["/api/tags"] = response
    assert (await health.check_ollama()).status.startswith("error:")
    assert await probes.OllamaProbe(health).get_models() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 302, 401, 503])
async def test_openrouter_status_and_authentication(http_probe: HttpFixture, status: int) -> None:
    health, responses, requests = http_probe
    responses["/models"] = httpx.Response(status)
    result = await health.check_openrouter()
    assert result.status == ("ok" if status < 400 else f"error: HTTP {status}")
    assert result.models == (["test-model"] if status < 400 else None)
    assert requests[0].headers["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 503])
async def test_tei_checks_both_endpoints(http_probe: HttpFixture, status: int) -> None:
    health, responses, requests = http_probe
    responses["/health"] = httpx.Response(status)
    result = await health.check_tei()
    assert set(result) == {"tei_embed", "tei_rerank"}
    assert all(check.status == "ok" for check in result.values()) == (status == 200)
    assert {request.url.host for request in requests} == {"embed", "rerank"}


@pytest.mark.asyncio
async def test_qdrant_admin_delegates_status() -> None:
    health = Mock(spec=probes.SystemHealthProbe)
    health.check_qdrant = AsyncMock(return_value=HealthCheckResult(status="error: offline"))
    assert await probes.QdrantInfo(health).get_status() == "error: offline"
    health.check_qdrant.assert_awaited_once()


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize(
    "operation", [probes.SystemHealthProbe().check_qdrant_sync, probes.QdrantInfo.fetch_collections]
)
def test_qdrant_client_closed_after_success_or_failure(
    monkeypatch: pytest.MonkeyPatch, fails: bool, operation: Callable[[], object]
) -> None:
    client = Mock()
    client.get_collections.return_value = SimpleNamespace(collections=[])

    if fails:
        client.get_collections.side_effect = RuntimeError("offline")

    monkeypatch.setattr(probes, "QdrantClient", Mock(return_value=client))

    if fails:
        with pytest.raises(RuntimeError, match="offline"):
            operation()
    else:
        operation()

    client.close.assert_called_once()


@pytest.mark.parametrize("named", [False, True])
def test_qdrant_collection_details_preserved(named: bool) -> None:
    vector = SimpleNamespace(size=768, distance=SimpleNamespace(value="Cosine"))
    params = SimpleNamespace(vectors={"dense": vector} if named else vector, on_disk_payload=True)
    client = Mock()
    client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="documents")])
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=params, hnsw_config=SimpleNamespace(m=16, ef_construct=100)),
        points_count=10,
        vectors_count=None,
        indexed_vectors_count=8,
        segments_count=1,
        status=SimpleNamespace(value="green"),
        optimizer_status="ok",
    )
    assert probes.QdrantInfo.read_collections(client) == [
        {
            "name": "documents",
            "points_count": 10,
            "vectors_count": 0,
            "indexed_vectors_count": 8,
            "segments_count": 1,
            "status": "green",
            "optimizer_status": "ok",
            "hnsw_m": 16,
            "hnsw_ef_construct": 100,
            "on_disk_payload": True,
            "vector_size": 768,
            "distance": "Cosine",
        }
    ]


@pytest.mark.asyncio
async def test_qdrant_collection_failure_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(asyncio, "to_thread", AsyncMock(side_effect=RuntimeError("offline")))
    assert await probes.QdrantInfo(probes.SystemHealthProbe()).get_collections() == []
    assert "Qdrant collection probe failed" in caplog.text
