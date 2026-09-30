"""Regression tests for Qdrant bulk upload (upload_to_qdrant / _upsert_with_semaphore).

Guards against passing unsupported kwargs to QdrantClient.upsert():
qdrant-client 1.11.3 rejects unknown arguments with
``AssertionError: Unknown arguments: [...]``, which made every outbox
``upsert_chunks`` entry fail and documents stay in 'indexing' forever.

The fake client mirrors the real ``QdrantClient.upsert`` signature exactly
(no ``**kwargs``), so any future reintroduction of a bogus kwarg breaks here.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from langchain.schema import Document

from config import settings
from infrastructure.repositories.vector.qdrant_ops import upload_to_qdrant


class StrictQdrantClient:
    """Mimics qdrant-client 1.11.3 QdrantClient.upsert argument contract."""

    def __init__(self) -> None:
        self.upsert_calls: list[dict] = []

    def upsert(
        self,
        collection_name: str,
        points: list,
        wait: bool = True,
        ordering: str | None = None,
        shard_key_selector: object | None = None,
    ) -> dict:
        # NOTE: no **kwargs — same as qdrant_client.QdrantClient.upsert's
        # effective contract after its ``assert len(kwargs) == 0``.
        self.upsert_calls.append(
            {
                "collection_name": collection_name,
                "points": list(points),
                "wait": wait,
                "ordering": ordering,
                "shard_key_selector": shard_key_selector,
            }
        )
        return {"status": {"acknowledged": True}}


class FakeEmbeddings:
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 1.0] for t in texts]


def _chunks(n: int) -> list[Document]:
    return [
        Document(page_content=f"chunk content {i}", metadata={"chunk_id": i, "source": "doc"})
        for i in range(n)
    ]


class TestUploadToQdrantUpsertContract:
    @pytest.mark.asyncio
    async def test_upsert_receives_only_supported_arguments(self):
        client = StrictQdrantClient()
        await upload_to_qdrant(_chunks(3), FakeEmbeddings(), client=client)

        assert len(client.upsert_calls) == 1
        call = client.upsert_calls[0]
        assert call["collection_name"] == settings.collection_name
        assert len(call["points"]) == 3
        assert sorted(p.id for p in call["points"]) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_upsert_with_write_semaphore(self):
        client = StrictQdrantClient()
        semaphore = asyncio.Semaphore(2)

        await upload_to_qdrant(_chunks(2), FakeEmbeddings(), client=client, write_semaphore=semaphore)

        assert len(client.upsert_calls) == 1
        assert len(client.upsert_calls[0]["points"]) == 2

    @pytest.mark.asyncio
    async def test_batched_upload_flushes_all_points(self):
        client = StrictQdrantClient()
        total = 501  # crosses the 500-point upsert batch boundary

        await upload_to_qdrant(_chunks(total), FakeEmbeddings(), client=client)

        flushed = [p for call in client.upsert_calls for p in call["points"]]
        assert len(flushed) == total
        assert sorted(p.id for p in flushed) == list(range(total))
