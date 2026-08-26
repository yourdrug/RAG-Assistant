"""Qdrant implementation of VectorStoreRepository."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from config import settings
from domain.entities.chunk import Chunk
from langchain.schema import Document as LCDocument
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from infrastructure.qdrant_ops import ensure_collection, upload_to_qdrant

if TYPE_CHECKING:
    from infrastructure.ml.client_registry import MLClientRegistry

log = logging.getLogger("default")


class QdrantVectorStoreRepository:
    def __init__(self, ml_clients: MLClientRegistry | None = None) -> None:
        self._ml_clients = ml_clients

    def _get_qdrant_client(self):
        if self._ml_clients is not None:
            return self._ml_clients.qdrant_client()
        from infrastructure.ml.factories import create_qdrant_client

        return create_qdrant_client()

    def _get_embeddings(self):
        if self._ml_clients is not None:
            return self._ml_clients.embeddings()
        from infrastructure.ml.factories import create_embeddings

        return create_embeddings()

    async def ensure_collection(self, vector_size: int, reset: bool = False) -> None:
        await asyncio.to_thread(ensure_collection, self._get_qdrant_client(), vector_size, reset=reset)

    async def upload_documents(self, chunks: list[Chunk]) -> None:
        lcdocs = [LCDocument(page_content=c.content, metadata=c.metadata) for c in chunks]
        await upload_to_qdrant(lcdocs, self._get_embeddings())

    async def delete_by_document_id(self, document_id: int) -> None:
        await asyncio.to_thread(
            self._get_qdrant_client().delete,
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
            ),
        )

    async def generate_embeddings(self, text: str) -> list[float]:
        return await self._get_embeddings().embed_query(text)

    async def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        client = self._get_qdrant_client()
        embeddings = self._get_embeddings()

        query_vector = await embeddings.embed_query(query)

        results = await asyncio.to_thread(
            client.search,
            collection_name=settings.collection_name,
            query_vector=query_vector,
            limit=k,
        )
        return [
            (
                Chunk(
                    content=doc.payload.get("page_content", ""),
                    metadata=doc.payload.get("metadata", {}),
                ),
                score,
            )
            for doc, score in results
        ]

    async def upsert_point(self, point_id: int, vector: list[float], payload: dict) -> None:
        """Upsert a single point with deterministic ID (chunk.id)."""
        client = self._get_qdrant_client()

        def _upsert() -> None:
            point = PointStruct(id=point_id, vector=vector, payload=payload)
            client.upsert(collection_name=settings.collection_name, points=[point])

        await asyncio.to_thread(_upsert)

    async def get_point_payload(self, point_id: int) -> dict | None:
        """Fetch a single point's payload by ID. Returns None if the point does not exist."""
        client = self._get_qdrant_client()

        def _get() -> dict | None:
            result = client.retrieve(
                collection_name=settings.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            return result[0].payload if result else None

        return await asyncio.to_thread(_get)

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
