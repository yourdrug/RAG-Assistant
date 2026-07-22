"""Qdrant implementation of VectorStoreRepository."""

from __future__ import annotations

import logging

from config import settings
from domain.entities.chunk import Chunk
from langchain.schema import Document as LCDocument
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from infrastructure.clients import get_embeddings
from infrastructure.qdrant_ops import ensure_collection, upload_to_qdrant

log = logging.getLogger("default")


class QdrantVectorStoreRepository:
    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self._store: QdrantVectorStore | None = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        return self._client

    def _get_store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore.from_existing_collection(
                embedding=get_embeddings(),
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                collection_name=settings.collection_name,
            )
        return self._store

    def ensure_collection(self, vector_size: int, reset: bool = False) -> None:
        ensure_collection(self._get_client(), vector_size, reset=reset)

    def upload_documents(self, chunks: list[Chunk]) -> None:
        lcdocs = [LCDocument(page_content=c.content, metadata=c.metadata) for c in chunks]
        upload_to_qdrant(lcdocs, get_embeddings())

    def delete_by_document_id(self, document_id: int) -> None:
        self._get_client().delete(
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )

    def as_retriever(self, search_kwargs: dict | None = None):
        return self._get_store().as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs or {"k": settings.retriever_top_k},
        )

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        results = self._get_store().similarity_search_with_score(query, k=k)
        return [(Chunk(content=doc.page_content, metadata=doc.metadata), score) for doc, score in results]
