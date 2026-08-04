"""Qdrant implementation of VectorStoreRepository."""

from __future__ import annotations

import asyncio
import logging

from config import settings
from domain.entities.chunk import Chunk
from langchain.schema import Document as LCDocument
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import FieldCondition, Filter, MatchValue

from infrastructure.clients import get_embeddings, get_qdrant_client, get_vector_store
from infrastructure.qdrant_ops import ensure_collection, upload_to_qdrant

log = logging.getLogger("default")


class QdrantVectorStoreRepository:
    def __init__(self) -> None:
        self._store: QdrantVectorStore | None = None

    def _get_store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = get_vector_store()
        return self._store

    async def ensure_collection(self, vector_size: int, reset: bool = False) -> None:
        await asyncio.to_thread(ensure_collection, get_qdrant_client(), vector_size, reset=reset)

    async def upload_documents(self, chunks: list[Chunk]) -> None:
        lcdocs = [LCDocument(page_content=c.content, metadata=c.metadata) for c in chunks]
        await asyncio.to_thread(upload_to_qdrant, lcdocs, get_embeddings())

    async def delete_by_document_id(self, document_id: int) -> None:
        await asyncio.to_thread(
            get_qdrant_client().delete,
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
            ),
        )

    async def generate_embeddings(self, text: str) -> list[float]:
        return await asyncio.to_thread(get_embeddings().embed_query, text)

    async def as_retriever(self, search_kwargs: dict | None = None):
        return self._get_store().as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs or {"k": settings.retriever_top_k},
        )

    async def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        results = await asyncio.to_thread(self._get_store().similarity_search_with_score, query, k=k)
        return [(Chunk(content=doc.page_content, metadata=doc.metadata), score) for doc, score in results]
