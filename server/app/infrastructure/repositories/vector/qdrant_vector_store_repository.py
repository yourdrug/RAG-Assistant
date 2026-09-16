"""Qdrant implementation of VectorStoreRepository."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from config import settings
from domain.entities.chunk import Chunk
from langchain.schema import Document as LCDocument
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from infrastructure.repositories.vector.qdrant_ops import ensure_collection, upload_to_qdrant
from infrastructure.resilience.retry import retry_on_transient

if TYPE_CHECKING:
    from infrastructure.ml.clients.client_registry import MLClientRegistry

log = logging.getLogger("default")

_BATCH_SIZE = 100


class QdrantVectorStoreRepository:
    def __init__(self, ml_clients: MLClientRegistry | None = None) -> None:
        self._ml_clients = ml_clients

    def _get_qdrant_client(self):
        if self._ml_clients is not None:
            return self._ml_clients.qdrant_client()
        from infrastructure.ml.clients.factories import create_qdrant_client

        return create_qdrant_client()

    def _get_embeddings(self):
        if self._ml_clients is not None:
            return self._ml_clients.embeddings()
        from infrastructure.ml.clients.factories import create_embeddings

        return create_embeddings()

    async def ensure_collection(self, vector_size: int, reset: bool = False) -> None:
        await asyncio.to_thread(ensure_collection, self._get_qdrant_client(), vector_size, reset=reset)

    async def upload_documents(
        self,
        chunks: list[Chunk],
        *,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        lcdocs = [LCDocument(page_content=c.content, metadata=c.metadata) for c in chunks]
        async with self._ml_clients.ingestion_semaphore:
            await upload_to_qdrant(
                lcdocs,
                self._get_embeddings(),
                client=self._get_qdrant_client(),
                write_semaphore=self._ml_clients.qdrant_write_semaphore,
                should_cancel=should_cancel,
            )

    @retry_on_transient
    async def delete_by_document_id(self, document_id: int) -> None:
        """Delete all vector points for a document."""
        client = self._get_qdrant_client()

        def _delete() -> None:
            client.delete(
                collection_name=settings.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
                ),
            )

        await asyncio.to_thread(_delete)
        log.info("Qdrant: deleted points for document_id=%d", document_id)

    async def generate_embeddings(self, text: str) -> list[float]:
        return await self._get_embeddings().embed_query(text)

    @retry_on_transient
    async def upsert_point(self, point_id: int, vector: list[float], payload: dict) -> None:
        """Upsert a single point with deterministic ID (chunk.id)."""
        client = self._get_qdrant_client()

        def _upsert() -> None:
            point = PointStruct(id=point_id, vector=vector, payload=payload)
            client.upsert(collection_name=settings.collection_name, points=[point])

        await asyncio.to_thread(_upsert)

    async def get_point_payload(self, point_id: int, access_filter: Filter) -> dict | None:
        """Fetch a single point's payload by ID with mandatory ACL enforcement.

        Raises ``RuntimeError`` if *access_filter* is not provided — all callers
        must pass an ACL filter to prevent cross-tenant data leaks.
        Returns None if the point does not exist or is outside ACL scope.
        """
        client = self._get_qdrant_client()
        combined_filter = Filter(
            must=[
                FieldCondition(key="id", match=MatchValue(value=point_id)),
                access_filter,
            ]
        )

        def _scroll() -> dict | None:
            results, _ = client.scroll(
                collection_name=settings.collection_name,
                scroll_filter=combined_filter,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            return results[0].payload if results else None

        return await asyncio.to_thread(_scroll)

    async def delete_by_ids(self, ids: list[int]) -> None:
        """Delete points by their IDs."""
        client = self._get_qdrant_client()

        def _delete() -> None:
            if not ids:
                return
            client.delete(
                collection_name=settings.collection_name,
                points_selector=ids,
            )

        await asyncio.to_thread(_delete)

    async def set_document_id_by_source(self, source: str, document_id: int) -> None:
        """Update document_id in metadata for all points with matching source."""
        client = self._get_qdrant_client()

        def _update() -> None:
            self._patch_metadata(
                client,
                Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]),
                {"document_id": document_id},
            )

        await asyncio.to_thread(_update)

    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> None:
        """Update filename in metadata for all points belonging to a document."""
        client = self._get_qdrant_client()

        def _update() -> None:
            self._patch_metadata(
                client,
                Filter(
                    must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
                ),
                {"filename": new_filename, "source": new_filename},
            )

        await asyncio.to_thread(_update)

    async def update_metadata_by_act_version(self, act_version_id: int, metadata_updates: dict) -> None:
        """Update metadata for all points matching a given act_version_id."""
        client = self._get_qdrant_client()

        def _update() -> None:
            self._patch_metadata(
                client,
                Filter(
                    must=[
                        FieldCondition(key="metadata.act_version_id", match=MatchValue(value=act_version_id))
                    ]
                ),
                metadata_updates,
            )

        await asyncio.to_thread(_update)

    @staticmethod
    def _patch_metadata(client, points_filter: Filter, metadata_updates: dict) -> None:
        """Read matching points, merge metadata_updates, and write back.

        Includes a 30-second safety timeout to prevent unbounded thread pool occupation.
        Raises ``RuntimeError`` if the timeout is exceeded before all points are processed.
        """
        import time

        _PATCH_TIMEOUT_SEC = 30
        t_start = time.monotonic()
        offset = None
        while True:
            if time.monotonic() - t_start > _PATCH_TIMEOUT_SEC:
                log.error(
                    "_patch_metadata: timed out after %ds — partial update possible",
                    _PATCH_TIMEOUT_SEC,
                )
                raise RuntimeError(
                    f"_patch_metadata timed out after {_PATCH_TIMEOUT_SEC}s. "
                    "Some points may not have been updated."
                )
            result, offset = client.scroll(
                collection_name=settings.collection_name,
                scroll_filter=points_filter,
                limit=_BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not result:
                break

            for point in result:
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                metadata.update(metadata_updates)
                payload["metadata"] = metadata
                client.set_payload(
                    collection_name=settings.collection_name,
                    payload=payload,
                    points=[point.id],
                )

            if offset is None:
                break
