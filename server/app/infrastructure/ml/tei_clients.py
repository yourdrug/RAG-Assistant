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

import httpx

log = logging.getLogger("default")

TEI_TIMEOUT = 600.0
RERANK_BATCH_SIZE = 8


class TEIEmbeddingsClient:
    """Embedding client that calls a TEI /embed endpoint over HTTP."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def embed_query(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=TEI_TIMEOUT) as client:
            r = await client.post(f"{self._base_url}/embed", json={"inputs": text})
            r.raise_for_status()
            return r.json()[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=TEI_TIMEOUT) as client:
            r = await client.post(f"{self._base_url}/embed", json={"inputs": texts})
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                short = repr(data)[:200]
                raise RuntimeError(f"TEI /embed returned unexpected type {type(data).__name__}: {short}")
            none_indices = [i for i, v in enumerate(data) if v is None]
            if none_indices:
                raise RuntimeError(
                    f"TEI /embed returned None for {len(none_indices)}/{len(data)} texts "
                    f"(indices: {none_indices[:10]}{'...' if len(none_indices) > 10 else ''})"
                )
            return data

    def embed_query_sync(self, text: str) -> list[float]:
        with httpx.Client(timeout=TEI_TIMEOUT) as client:
            r = client.post(f"{self._base_url}/embed", json={"inputs": text})
            r.raise_for_status()
            return r.json()[0]


class TEIRerankerClient:
    """Reranking client that calls a TEI /rerank endpoint over HTTP."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        texts = [p[1] for p in pairs]
        payload = {"query": query, "texts": texts}
        async with httpx.AsyncClient(timeout=TEI_TIMEOUT) as client:
            r = await client.post(f"{self._base_url}/rerank", json=payload)
            r.raise_for_status()
            data = r.json()
            return [item["score"] for item in data]

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
            return [item["score"] for item in data]
