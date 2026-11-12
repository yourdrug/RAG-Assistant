"""Qdrant adapter implementing vector index writes and ACL-filtered payload reads."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain.schema import Document as LCDocument
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from config import VectorStoreSection, process_section
from domain.rag.evidence.context import content_with_parent_scope
from infrastructure.corpus.snapshot import collection_for, pin_corpus_call
from infrastructure.ml.clients.factories import create_embeddings, create_qdrant_client
from infrastructure.repositories.vector.embedding_identity import ensure_embedding_identity
from infrastructure.repositories.vector.qdrant_ops import ensure_collection, upload_to_qdrant
from infrastructure.repositories.vector.sparse_vectors import point_vectors
from infrastructure.resilience.retry import retry_on_transient


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.ports.ml_clients import MLClientPort
    from domain.knowledge.chunk import Chunk
    from infrastructure.repositories.vector.client_types import DocumentEmbeddingClient, VectorClient

log = logging.getLogger("default")


class QdrantVectorStoreRepository:
    def __init__(self, ml_clients: MLClientPort | None = None) -> None:
        self._ml_clients = ml_clients

    def get_qdrant_client(self) -> VectorClient:
        if self._ml_clients is not None:
            # The application port intentionally treats provider clients as opaque.
            return cast("VectorClient", self._ml_clients.qdrant_client())

        return create_qdrant_client()

    def get_embeddings(self) -> DocumentEmbeddingClient:
        if self._ml_clients is not None:
            return cast("DocumentEmbeddingClient", self._ml_clients.embeddings())

        return cast("DocumentEmbeddingClient", create_embeddings())

    @pin_corpus_call
    async def ensure_collection(self, vector_size: int, *, reset: bool = False) -> None:
        await asyncio.to_thread(ensure_collection, self.get_qdrant_client(), vector_size, reset=reset)

    @pin_corpus_call
    async def upload_documents(
        self,
        chunks: list[Chunk],
        *,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        if self._ml_clients is None:
            msg = "ml_clients not initialized — cannot upload documents"
            raise ValueError(msg)

        lcdocs = [LCDocument(page_content=c.content, metadata=c.metadata) for c in chunks]
        async with self._ml_clients.ingestion_semaphore:
            await upload_to_qdrant(
                lcdocs,
                self.get_embeddings(),
                client=self.get_qdrant_client(),
                write_semaphore=self._ml_clients.qdrant_write_semaphore,
                should_cancel=should_cancel,
            )

    @pin_corpus_call
    @retry_on_transient
    async def delete_by_document_id(self, document_id: int) -> None:
        """Delete all vector points for a document."""
        client = self.get_qdrant_client()

        def delete() -> None:
            document_filter = Filter(
                must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
            )
            client.delete(
                collection_name=collection_for(client),
                points_selector=document_filter,
                wait=True,
            )
            remaining, _ = client.scroll(
                collection_name=collection_for(client),
                scroll_filter=document_filter,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )

            if remaining:
                message = f"Qdrant still contains points for deleted document_id={document_id}"
                raise RuntimeError(message)

        await asyncio.to_thread(delete)
        log.info("Qdrant: deleted points for document_id=%d", document_id)

    async def generate_embeddings(self, text: str) -> list[float]:
        return await self.get_embeddings().embed_query(text)

    @pin_corpus_call
    @retry_on_transient
    async def upsert_point(self, point_id: int, vector: list[float], payload: dict[str, Any]) -> None:
        """Upsert a single point with deterministic ID (chunk.id)."""
        client = self.get_qdrant_client()

        def upsert() -> None:
            identity = payload.get("metadata", {}).get("embedding_model")

            if not isinstance(identity, str) or not identity:
                msg = "Vector payload must declare its originating embedding_model"
                raise ValueError(msg)

            ensure_embedding_identity(client, identity)
            # Sparse text must match bulk upload (upload_document_batches), which
            # indexes the chunk together with its parent scope.
            scoped_text = content_with_parent_scope(
                LCDocument(page_content=payload["page_content"], metadata=payload.get("metadata", {}))
            )
            point = PointStruct(id=point_id, vector=point_vectors(scoped_text, vector), payload=payload)
            client.upsert(collection_name=process_section(VectorStoreSection).collection_name, points=[point])

        await asyncio.to_thread(upsert)

    @pin_corpus_call
    @retry_on_transient
    async def get_point_payload(self, point_id: int, access_filter: Filter) -> dict[str, Any] | None:
        """Fetch a single point's payload by ID with mandatory ACL enforcement.

        Raises ``RuntimeError`` if *access_filter* is not provided — all callers
        must pass an ACL filter to prevent cross-tenant data leaks.
        Returns None if the point does not exist or is outside ACL scope.
        """
        client = self.get_qdrant_client()
        combined_filter = Filter(
            must=[
                FieldCondition(key="id", match=MatchValue(value=point_id)),
                access_filter,
            ]
        )

        def scroll() -> dict[str, Any] | None:
            results, _ = client.scroll(
                collection_name=process_section(VectorStoreSection).collection_name,
                scroll_filter=combined_filter,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            return results[0].payload if results else None

        return await asyncio.to_thread(scroll)

    @pin_corpus_call
    @retry_on_transient
    async def delete_by_ids(self, ids: list[int]) -> None:
        """Delete points by their IDs."""
        client = self.get_qdrant_client()

        def delete() -> None:
            if not ids:
                return

            client.delete(
                collection_name=process_section(VectorStoreSection).collection_name,
                points_selector=ids,
            )

        await asyncio.to_thread(delete)

    @pin_corpus_call
    @retry_on_transient
    async def set_document_id_by_source(self, source: str, document_id: int) -> None:
        """Update document_id in metadata for all points with matching source."""
        client = self.get_qdrant_client()

        def update() -> None:
            self.patch_metadata(
                client,
                Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]),
                {"document_id": document_id},
            )

        await asyncio.to_thread(update)

    @pin_corpus_call
    @retry_on_transient
    async def update_filename_by_document_id(self, document_id: int, new_filename: str) -> None:
        """Update filename in metadata for all points belonging to a document."""
        client = self.get_qdrant_client()

        def update() -> None:
            self.patch_metadata(
                client,
                Filter(
                    must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
                ),
                {"filename": new_filename, "source": new_filename},
            )

        await asyncio.to_thread(update)

    @pin_corpus_call
    @retry_on_transient
    async def update_metadata_by_act_version(
        self, act_version_id: int, metadata_updates: dict[str, Any]
    ) -> None:
        """Update metadata for all points matching a given act_version_id."""
        client = self.get_qdrant_client()

        def update() -> None:
            self.patch_metadata(
                client,
                Filter(
                    must=[
                        FieldCondition(key="metadata.act_version_id", match=MatchValue(value=act_version_id))
                    ]
                ),
                metadata_updates,
            )

        await asyncio.to_thread(update)

    @staticmethod
    def patch_metadata(client: VectorClient, points_filter: Filter, metadata_updates: dict[str, Any]) -> None:
        """Patch only selected metadata keys without reading or replacing vectors."""
        client.set_payload(
            collection_name=process_section(VectorStoreSection).collection_name,
            payload=metadata_updates,
            points=points_filter,
            key="metadata",
            wait=True,
        )
