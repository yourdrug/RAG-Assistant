"""Qdrant implementation of VectorStoreRepository."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from config import settings
from domain.entities.chunk import Chunk
from langchain.schema import Document as LCDocument
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from infrastructure.qdrant_ops import ensure_collection, upload_to_qdrant

if TYPE_CHECKING:
    from infrastructure.ml.client_registry import MLClientRegistry

log = logging.getLogger("default")


class QdrantVectorStoreRepository:
    def __init__(self, ml_clients: MLClientRegistry | None = None) -> None:
        self._store: QdrantVectorStore | None = None
        self._ml_clients = ml_clients

    def _get_store(self) -> QdrantVectorStore:
        if self._store is None:
            if self._ml_clients is not None:
                self._store = self._ml_clients.vector_store()
            else:
                from infrastructure.ml.factories import create_embeddings, create_vector_store

                self._store = create_vector_store(create_embeddings())
        return self._store

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
        await asyncio.to_thread(upload_to_qdrant, lcdocs, self._get_embeddings())

    async def delete_by_document_id(self, document_id: int) -> None:
        await asyncio.to_thread(
            self._get_qdrant_client().delete,
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
            ),
        )

    async def generate_embeddings(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_embeddings().embed_query, text)

    async def as_retriever(self, search_kwargs: dict | None = None):
        return self._get_store().as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs or {"k": settings.retriever_top_k},
        )

    async def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        results = await asyncio.to_thread(self._get_store().similarity_search_with_score, query, k=k)
        return [(Chunk(content=doc.page_content, metadata=doc.metadata), score) for doc, score in results]

    async def upsert_point(self, point_id: int, vector: list[float], payload: dict) -> None:
        """Upsert a single point with deterministic ID (chunk.id)."""
        client = self._get_qdrant_client()

        def _upsert() -> None:
            point = PointStruct(id=point_id, vector=vector, payload=payload)
            client.upsert(collection_name=settings.collection_name, points=[point])

        await asyncio.to_thread(_upsert)

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
