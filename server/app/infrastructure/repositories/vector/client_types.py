"""Provider types narrowed at the opaque ML-client adapter boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from application.ports.ml_clients import EmbeddingClientPort


if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from infrastructure.corpus.snapshot import BoundQdrantClient


type VectorClient = QdrantClient | BoundQdrantClient


class DocumentEmbeddingClient(EmbeddingClientPort, Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
