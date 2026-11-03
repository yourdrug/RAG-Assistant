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
import threading
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from langchain.schema import Document
import pytest
from qdrant_client import QdrantClient

from config import settings
from infrastructure.repositories.vector import qdrant_ops
from infrastructure.repositories.vector.qdrant_ops import upload_to_qdrant
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME


if TYPE_CHECKING:
    from qdrant_client.models import PointStruct


class StrictQdrantClient:
    """Mimics qdrant-client 1.11.3 QdrantClient.upsert argument contract."""

    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, Any]] = []

    def upsert(
        self,
        collection_name: str,
        points: list[PointStruct],
        wait: bool = True,
        ordering: str | None = None,
        shard_key_selector: object | None = None,
    ) -> dict[str, Any]:
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
    async def embed_query(self, text: str) -> list[float]:
        return self.embed_query_sync(text)

    def embed_query_sync(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 1.0] for t in texts]


@pytest.fixture(autouse=True)
def match_fake_embedding_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embed_dim", 3)
    monkeypatch.setattr(
        "infrastructure.repositories.vector.qdrant_ops.ensure_embedding_identity",
        lambda *_args: "test:embedding:v1",
    )


def _chunks(n: int) -> list[Document]:
    return [
        Document(page_content=f"chunk content {i}", metadata={"chunk_id": i, "source": "doc"})
        for i in range(n)
    ]


class TestUploadToQdrantUpsertContract:
    @pytest.mark.asyncio
    async def test_upsert_receives_only_supported_arguments(self) -> None:
        client = StrictQdrantClient()
        await upload_to_qdrant(
            _chunks(3), FakeEmbeddings(), client=MagicMock(spec=QdrantClient, wraps=client)
        )

        assert len(client.upsert_calls) == 1
        call = client.upsert_calls[0]
        assert call["collection_name"] == settings.collection_name
        assert len(call["points"]) == 3
        assert sorted(p.id for p in call["points"]) == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_upsert_with_write_semaphore(self) -> None:
        client = StrictQdrantClient()
        semaphore = asyncio.Semaphore(2)

        await upload_to_qdrant(
            _chunks(2),
            FakeEmbeddings(),
            client=MagicMock(spec=QdrantClient, wraps=client),
            write_semaphore=semaphore,
        )

        assert len(client.upsert_calls) == 1
        assert len(client.upsert_calls[0]["points"]) == 2

    @pytest.mark.asyncio
    async def test_batched_upload_flushes_all_points(self) -> None:
        client = StrictQdrantClient()
        total = 501  # crosses the 500-point upsert batch boundary

        await upload_to_qdrant(
            _chunks(total), FakeEmbeddings(), client=MagicMock(spec=QdrantClient, wraps=client)
        )

        flushed = [p for call in client.upsert_calls for p in call["points"]]
        assert len(flushed) == total
        assert sorted(p.id for p in flushed) == list(range(total))


@pytest.mark.asyncio
async def test_sparse_encoding_runs_off_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread = threading.get_ident()
    threads = []
    original = qdrant_ops.build_points

    def encode(*args: Any) -> list[PointStruct]:
        threads.append(threading.get_ident())
        return original(*args)

    monkeypatch.setattr(qdrant_ops, "build_points", encode)
    client = StrictQdrantClient()
    await upload_to_qdrant(_chunks(3), FakeEmbeddings(), client=MagicMock(spec=QdrantClient, wraps=client))
    assert threads
    assert all(thread != loop_thread for thread in threads)

    for point in client.upsert_calls[0]["points"]:
        assert set(point.vector) == {"", SPARSE_VECTOR_NAME}
        assert point.vector[SPARSE_VECTOR_NAME].indices
