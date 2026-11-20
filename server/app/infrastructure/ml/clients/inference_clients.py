"""HTTP clients for embedding and reranking endpoints.

Embeddings use the OpenAI-compatible ``POST {base}/embeddings`` (TEI ``/v1``, DeepInfra
``/v1/openai``, vLLM, OpenAI, Ollama, ...). Reranking has no common standard, so the
client speaks one of ``RerankApiFormat``; every format returns scores in ``[0, 1]``.

Each client exposes async methods and sync variants (``_sync`` suffix) for benchmarks.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, cast

import httpx

from application.runtime.enums import RerankApiFormat
from domain.shared.exceptions import BusinessRuleViolationError
from infrastructure.ml.clients.managed_http import ManagedHTTPPool, retrying_sync_transport
from infrastructure.ml.usage_capture import record_inference_response
from infrastructure.telemetry.tracing import paid_call


if TYPE_CHECKING:
    from contextlib import AbstractContextManager


log = logging.getLogger("default")

INFERENCE_TIMEOUT = 60.0


class InferenceResponseError(RuntimeError):
    """The endpoint returned a malformed embedding or reranking response."""


def bearer_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def probability(logit: float) -> float:
    """Overflow-safe float64 sigmoid that keeps strong matches distinguishable."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))

    exp = math.exp(logit)
    return exp / (1.0 + exp)


class EmbeddingsClient(ManagedHTTPPool):
    """Embedding client for an OpenAI-compatible ``/embeddings`` endpoint."""

    embedding_identity: str

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        pool_limits: httpx.Limits | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        limits = pool_limits or httpx.Limits(max_connections=20, max_keepalive_connections=10)
        super().__init__(INFERENCE_TIMEOUT, limits)

    def request(self, inputs: str | list[str]) -> dict[str, Any]:
        return {
            "url": f"{self._base_url}/embeddings",
            "headers": bearer_headers(self._api_key),
            "json": {"input": inputs, "model": self._model, "encoding_format": "float"},
        }

    def vectors(self, response: httpx.Response, count: int) -> list[list[float]]:
        response.raise_for_status()
        payload = response.json()
        record_inference_response(payload, model=self._model, operation="embedding")
        data = payload.get("data") if isinstance(payload, dict) else None

        if not isinstance(data, list) or len(data) != count:
            message = f"Embedding endpoint returned {type(data).__name__} instead of {count} vectors"
            raise InferenceResponseError(message)

        ordered = sorted(data, key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0)
        vectors = [item.get("embedding") if isinstance(item, dict) else None for item in ordered]

        if any(not isinstance(vector, list) for vector in vectors):
            message = "Embedding endpoint returned an item without a vector"
            raise InferenceResponseError(message)

        return cast("list[list[float]]", vectors)

    def traced(self, count: int) -> AbstractContextManager[object]:
        return paid_call("embeddings", self._model, base_url=self._base_url, **{"rag.batch_size": count})

    async def embed_query(self, text: str) -> list[float]:
        with self.traced(1):
            return self.vectors(await self.post(**self.request(text)), 1)[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        with self.traced(len(texts)):
            return self.vectors(await self.post(**self.request(texts)), len(texts))

    def embed_query_sync(self, text: str) -> list[float]:
        with (
            self.traced(1),
            httpx.Client(
                timeout=INFERENCE_TIMEOUT, transport=retrying_sync_transport(), follow_redirects=False
            ) as client,
        ):
            return self.vectors(client.post(**self.request(text)), 1)[0]


def indexed_scores(items: object, count: int, score_key: str, *, logits: bool) -> list[float]:
    """Map ranked ``{index, <score_key>}`` items back to input order."""
    if not isinstance(items, list):
        message = "Rerank endpoint returned no result list"
        raise InferenceResponseError(message)

    scores: dict[int, float] = {}

    for item in items:
        index = item.get("index") if isinstance(item, dict) else None

        if type(index) is not int or not 0 <= index < count or index in scores:
            message = "Invalid or duplicate rerank index"
            raise InferenceResponseError(message)

        value = float(item[score_key])

        if not math.isfinite(value):
            message = "Non-finite rerank score"
            raise InferenceResponseError(message)

        scores[index] = probability(value) if logits else value

    if len(scores) != count:
        message = "Incomplete rerank response"
        raise InferenceResponseError(message)

    return [scores[index] for index in range(count)]


class RerankerClient(ManagedHTTPPool):
    """Reranking client; ``.predict(pairs)`` returns one ``[0, 1]`` score per pair."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_format: RerankApiFormat | str,
        api_key: str = "",
        pool_limits: httpx.Limits | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._format = RerankApiFormat(api_format)
        self._api_key = api_key
        limits = pool_limits or httpx.Limits(max_connections=10, max_keepalive_connections=5)
        super().__init__(INFERENCE_TIMEOUT, limits)

    def request(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        if any(pair[0] != pairs[0][0] for pair in pairs):
            message = "Reranking pairs must share one query"
            raise BusinessRuleViolationError(message)

        query = pairs[0][0]
        documents = [pair[1] for pair in pairs]
        headers = bearer_headers(self._api_key)

        if self._format == RerankApiFormat.COHERE:
            body: dict[str, Any] = {"model": self._model, "query": query, "documents": documents}
            return {"url": f"{self._base_url}/rerank", "headers": headers, "json": body}

        if self._format == RerankApiFormat.TEI:
            # Logits are converted locally: TEI applies sigmoid in float32, which
            # collapses strong matches to 1.0 and makes their order arbitrary.
            body = {"query": query, "texts": documents, "raw_scores": True}
            return {"url": f"{self._base_url}/rerank", "headers": headers, "json": body}

        body = {"queries": [query], "documents": documents}
        return {"url": f"{self._base_url}/inference/{self._model}", "headers": headers, "json": body}

    def scores(self, response: httpx.Response, count: int) -> list[float]:
        response.raise_for_status()
        payload = response.json()

        if self._format == RerankApiFormat.TEI:
            return indexed_scores(payload, count, "score", logits=True)

        record_inference_response(payload, model=self._model, operation="reranking")

        if self._format == RerankApiFormat.COHERE:
            results = payload.get("results") if isinstance(payload, dict) else None
            return indexed_scores(results, count, "relevance_score", logits=False)

        scores = payload.get("scores") if isinstance(payload, dict) else None

        if not isinstance(scores, list) or len(scores) != count:
            message = "Rerank endpoint returned an unexpected score list"
            raise InferenceResponseError(message)

        return [float(score) for score in scores]

    def traced(self, count: int) -> AbstractContextManager[object]:
        return paid_call("rerank", self._model, base_url=self._base_url, **{"rag.batch_size": count})

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        with self.traced(len(pairs)):
            return self.scores(await self.post(**self.request(pairs)), len(pairs))

    def predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        with (
            self.traced(len(pairs)),
            httpx.Client(
                timeout=INFERENCE_TIMEOUT, transport=retrying_sync_transport(), follow_redirects=False
            ) as client,
        ):
            return self.scores(client.post(**self.request(pairs)), len(pairs))
