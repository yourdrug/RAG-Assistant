"""Embedding and rerank pools survive transient upstream failures without resending processed requests."""

import asyncio

import httpx
import pytest

from infrastructure.ml.clients import managed_http
from infrastructure.ml.clients.inference_clients import EmbeddingsClient


def embeddings_client(handler: httpx.MockTransport) -> EmbeddingsClient:
    client = EmbeddingsClient("https://embed.example/v1", "bge-m3", api_key="sk-test")
    client._client = httpx.AsyncClient(transport=handler)
    return client


def vector_response() -> httpx.Response:
    return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


@pytest.mark.asyncio
async def test_rate_limited_request_is_repeated_after_retry_after(waits: list[float]) -> None:
    responses = [httpx.Response(429, headers={"Retry-After": "3"}), vector_response()]
    client = embeddings_client(httpx.MockTransport(lambda _request: responses.pop(0)))

    try:
        assert await client.embed_query("q") == [0.1, 0.2]
    finally:
        await client.close()

    assert waits == [3.0]


@pytest.mark.asyncio
async def test_persistent_rate_limit_still_fails_after_bounded_retries(waits: list[float]) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "600"})

    client = embeddings_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.embed_query("q")
    finally:
        await client.close()

    assert len(calls) == managed_http.RATE_LIMIT_RETRIES + 1
    # A long Retry-After is capped so a run never stalls on one request.
    assert waits == [managed_http.MAX_RATE_LIMIT_WAIT_SEC] * managed_http.RATE_LIMIT_RETRIES


@pytest.mark.asyncio
async def test_server_errors_are_not_repeated(waits: list[float]) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    client = embeddings_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.embed_query("q")
    finally:
        await client.close()

    assert len(calls) == 1
    assert waits == []


def test_pools_retry_failed_connections_only() -> None:
    # httpcore repeats only connection setup (refused, TLS handshake, connect timeout).
    transport = managed_http.retrying_transport(httpx.Limits(max_connections=1))

    assert transport._pool._retries == managed_http.CONNECT_RETRIES
    assert managed_http.retrying_sync_transport()._pool._retries == managed_http.CONNECT_RETRIES
