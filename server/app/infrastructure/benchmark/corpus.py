"""Pin benchmark work and identify reports with the corpus actually used."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from config import settings
from infrastructure.corpus.snapshot import CorpusSnapshot
from infrastructure.ml.clients.client_registry import MLClientRegistry


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from application.ports.ml_clients import MLClientPort


@asynccontextmanager
async def benchmark_corpus(ml_clients: MLClientPort | None) -> AsyncIterator[CorpusSnapshot | None]:
    if isinstance(ml_clients, MLClientRegistry):
        async with ml_clients.corpus_operation():
            yield getattr(ml_clients.qdrant_client(), "snapshot", None)
    else:
        yield None


def corpus_metadata(snapshot: CorpusSnapshot | None) -> dict[str, Any]:
    if isinstance(snapshot, CorpusSnapshot):
        return {
            "collection_name": snapshot.physical_collection,
            "qdrant_url": snapshot.endpoint,
            "corpus_generation": snapshot.generation,
            "embedding_identity": snapshot.embedding_identity,
            "corpus_namespace": snapshot.namespace,
        }

    return {"collection_name": settings.collection_name, "qdrant_url": settings.qdrant_url}


@asynccontextmanager
async def benchmark_scope(ml_clients: MLClientPort | None) -> AsyncIterator[dict[str, Any]]:
    async with benchmark_corpus(ml_clients) as corpus:
        yield corpus_metadata(corpus)
