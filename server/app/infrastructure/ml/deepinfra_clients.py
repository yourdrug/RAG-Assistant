"""HTTP clients for DeepInfra managed embedding and reranking APIs.

DeepInfra provides:
  - Embeddings: OpenAI-compatible endpoint (/v1/openai/embeddings)
  - Reranking: native endpoint (/v1/inference/{model})

Each client exposes both async and sync methods matching the same interface
as the TEI clients, so the rest of the codebase is provider-agnostic.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger("default")

DEEPINFRA_TIMEOUT = 600.0


class DeepInfraEmbeddingsClient:
    """Embedding client using DeepInfra OpenAI-compatible /embeddings endpoint."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def embed_query(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=DEEPINFRA_TIMEOUT) as client:
            r = await client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(),
                json={"input": text, "model": self._model, "encoding_format": "float"},
            )
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=DEEPINFRA_TIMEOUT) as client:
            r = await client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(),
                json={"input": texts, "model": self._model, "encoding_format": "float"},
            )
            r.raise_for_status()
            data = r.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def embed_query_sync(self, text: str) -> list[float]:
        with httpx.Client(timeout=DEEPINFRA_TIMEOUT) as client:
            r = client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(),
                json={"input": text, "model": self._model, "encoding_format": "float"},
            )
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]


class DeepInfraRerankerClient:
    """Reranking client using DeepInfra native /inference/{model} endpoint.

    Interface matches TEIRerankerClient: .predict(pairs) and .predict_sync(pairs).
    """

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        documents = [p[1] for p in pairs]
        payload = {"queries": [query], "documents": documents}
        async with httpx.AsyncClient(timeout=DEEPINFRA_TIMEOUT) as client:
            r = await client.post(
                f"{self._base_url}/inference/{self._model}",
                headers=self._headers(),
                json=payload,
            )
            r.raise_for_status()
            return r.json()["scores"]

    def predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        documents = [p[1] for p in pairs]
        payload = {"queries": [query], "documents": documents}
        with httpx.Client(timeout=DEEPINFRA_TIMEOUT) as client:
            r = client.post(
                f"{self._base_url}/inference/{self._model}",
                headers=self._headers(),
                json=payload,
            )
            r.raise_for_status()
            return r.json()["scores"]
