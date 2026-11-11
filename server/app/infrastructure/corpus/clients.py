"""Create deployment-approved Qdrant clients and probe the active corpus."""

from urllib.parse import urlsplit

from qdrant_client import QdrantClient
from qdrant_client.models import Modifier

from config import MLProviderSection, VectorStoreSection, process_section
from domain.shared.exceptions import ValidationError
from infrastructure.corpus.snapshot import CorpusSnapshot
from infrastructure.corpus.validation import validate_corpus_binding
from infrastructure.ml.config.endpoint_policy import endpoint_origin
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME


def raw_corpus_client(endpoint: str) -> QdrantClient:
    vector_store_settings = process_section(VectorStoreSection)

    allowed = {
        endpoint_origin(value) for value in process_section(MLProviderSection).ml_internal_allowed_origins
    }

    if endpoint_origin(endpoint) not in allowed or urlsplit(endpoint).path not in {"", "/"}:
        message = "Qdrant origin must be deployment-approved before credentials are used"
        raise ValidationError(message)

    return QdrantClient(
        url=endpoint,
        api_key=vector_store_settings.qdrant_api_key,
        timeout=vector_store_settings.qdrant_timeout,
        follow_redirects=False,
    )


def validate_corpus_endpoint(endpoint: str, current: str) -> None:
    allowed = {endpoint_origin(current)} | {
        endpoint_origin(v) for v in process_section(MLProviderSection).ml_internal_allowed_origins
    }

    if endpoint_origin(endpoint) not in allowed or urlsplit(endpoint).path not in {"", "/"}:
        message = "Qdrant origin is not deployment-approved"
        raise ValidationError(message)


def probe_active_corpus(client: QdrantClient, snapshot: CorpusSnapshot, *, completed: bool) -> None:
    ml_provider_settings = process_section(MLProviderSection)

    names = {item.name for item in client.get_collections().collections}
    physical = snapshot.physical_collection

    if physical not in names:
        if completed:
            message = "Completed PostgreSQL chunks have no active Qdrant collection"
            raise ValidationError(message)

        return  # Fresh empty deployment; ingestion owns collection creation.

    info = client.get_collection(physical)
    sparse = info.config.params.sparse_vectors or {}

    if SPARSE_VECTOR_NAME not in sparse or sparse[SPARSE_VECTOR_NAME].modifier != Modifier.IDF:
        message = "Active corpus has no compatible BM25 sparse index"
        raise ValidationError(message)

    if not info.points_count and physical + "__embedding_identity" not in names:
        if completed or getattr(info.config.params.vectors, "size", None) != ml_provider_settings.embed_dim:
            message = "Active corpus has no compatible immutable identity"
            raise ValidationError(message)

        return

    validate_corpus_binding(
        client,
        snapshot.endpoint,
        physical,
        snapshot.embedding_identity,
        ml_provider_settings.embed_dim,
    )
