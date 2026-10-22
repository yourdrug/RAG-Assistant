"""Pin benchmark work and identify reports with the corpus actually used."""

from contextlib import asynccontextmanager

from config import settings
from infrastructure.corpus.snapshot import CorpusSnapshot
from infrastructure.ml.clients.client_registry import MLClientRegistry


@asynccontextmanager
async def benchmark_corpus(ml_clients):
    if isinstance(ml_clients, MLClientRegistry):
        async with ml_clients.corpus_operation():
            yield getattr(ml_clients.qdrant_client(), 'snapshot', None)
    else:
        yield None


def corpus_metadata(snapshot) -> dict:
    if isinstance(snapshot, CorpusSnapshot):
        return {
            'collection_name': snapshot.physical_collection,
            'qdrant_url': snapshot.endpoint,
            'corpus_generation': snapshot.generation,
            'embedding_identity': snapshot.embedding_identity,
            'corpus_namespace': snapshot.namespace,
        }

    return {'collection_name': settings.collection_name, 'qdrant_url': settings.qdrant_url}
