"""HTTP clients for Text Embeddings Inference (TEI) services.

TEI is a separate service that hosts embedding and reranking models.
The API pod calls TEI synchronously over HTTP — the same pattern as
calling Qdrant or Ollama (sync request/response, not async worker).

Each client exposes both async and sync methods:
  - async: for use inside FastAPI / async code paths
  - sync  (_sync suffix): for benchmark and other sync code paths
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast

import httpx

from infrastructure.ml.clients.managed_http import ManagedHTTPPool


log = logging.getLogger("default")

TEI_TIMEOUT = 60.0
MAX_ERROR_INDICES = 10


class TEIResponseError(RuntimeError):
    """TEI returned an invalid embedding response."""


class TEIEmbeddingsClient(ManagedHTTPPool):
    """Embedding client that calls a TEI /embed endpoint over HTTP."""

    embedding_identity: str

    def __init__(self, base_url: str, pool_limits: httpx.Limits | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        limits = pool_limits or httpx.Limits(max_connections=20, max_keepalive_connections=10)
        super().__init__(TEI_TIMEOUT, limits)

    async def embed_query(self, text: str) -> list[float]:
        r = await self.post(f"{self._base_url}/embed", json={"inputs": text})
        r.raise_for_status()
        return cast("list[float]", r.json()[0])

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        r = await self.post(f"{self._base_url}/embed", json={"inputs": texts})
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, list):
            short = repr(data)[:200]
            message = f"TEI /embed returned unexpected type {type(data).__name__}: {short}"
            raise TEIResponseError(message)

        none_indices = [i for i, v in enumerate(data) if v is None]

        if none_indices:
            preview = none_indices[:MAX_ERROR_INDICES]
            suffix = "..." if len(none_indices) > MAX_ERROR_INDICES else ""
            message = (
                f"TEI /embed returned None for {len(none_indices)}/{len(data)} texts "
                f"(indices: {preview}{suffix})"
            )
            raise TEIResponseError(message)

        return cast("list[list[float]]", data)

    def embed_query_sync(self, text: str) -> list[float]:
        with httpx.Client(timeout=TEI_TIMEOUT) as client:
            r = client.post(f"{self._base_url}/embed", json={"inputs": text})
            r.raise_for_status()
            return cast("list[float]", r.json()[0])


class TEIRerankerClient(ManagedHTTPPool):
    """Reranking client that calls a TEI /rerank endpoint over HTTP."""

    def __init__(self, base_url: str, pool_limits: httpx.Limits | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        limits = pool_limits or httpx.Limits(max_connections=10, max_keepalive_connections=5)
        super().__init__(TEI_TIMEOUT, limits)

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        query = pairs[0][0]
        texts = [p[1] for p in pairs]
        payload = {"query": query, "texts": texts}
        r = await self.post(f"{self._base_url}/rerank", json=payload)
        r.raise_for_status()
        data = r.json()
        return self.ordered_scores(data, len(texts))

    def predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []

        query = pairs[0][0]
        texts = [p[1] for p in pairs]
        payload = {"query": query, "texts": texts}
        with httpx.Client(timeout=TEI_TIMEOUT) as client:
            r = client.post(f"{self._base_url}/rerank", json=payload)
            r.raise_for_status()
            data = r.json()
            return self.ordered_scores(data, len(texts))

    @staticmethod
    def ordered_scores(data: list[dict[str, Any]], count: int) -> list[float]:
        """TEI ranks results; its index identifies the original input text."""
        scores: dict[int, float] = {}

        for item in data:
            index = item.get("index")

            if type(index) is not int or not 0 <= index < count or index in scores:
                message = "Invalid or duplicate TEI rerank index"
                raise ValueError(message)

            score = float(item["score"])

            if not math.isfinite(score):
                message = "Non-finite TEI rerank score"
                raise ValueError(message)

            scores[index] = score

        if len(scores) != count:
            message = "Incomplete TEI rerank response"
            raise ValueError(message)

        return [scores[index] for index in range(count)]
