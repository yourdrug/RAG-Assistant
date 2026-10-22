"""Create deployment-approved Qdrant clients and probe the active corpus."""

from infrastructure.corpus.snapshot import CorpusSnapshot
from urllib.parse import urlsplit
from qdrant_client import QdrantClient
from qdrant_client.models import Modifier
from config import settings
from domain.shared.exceptions import ValidationError
from infrastructure.corpus.validation import validate_corpus_binding
from infrastructure.ml.config.endpoint_policy import endpoint_origin
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME


def raw_corpus_client(endpoint: str) -> QdrantClient:
    allowed = {endpoint_origin(value) for value in settings.ml_internal_allowed_origins}

    if endpoint_origin(endpoint) not in allowed or urlsplit(endpoint).path not in {"", "/"}:
        raise ValidationError("Qdrant origin must be deployment-approved before credentials are used")

    return QdrantClient(
        url=endpoint, api_key=settings.qdrant_api_key, timeout=settings.qdrant_timeout, follow_redirects=False
    )


def validate_corpus_endpoint(endpoint: str, current: str) -> None:
    allowed = {endpoint_origin(current)} | {endpoint_origin(v) for v in settings.ml_internal_allowed_origins}

    if endpoint_origin(endpoint) not in allowed or urlsplit(endpoint).path not in {"", "/"}:
        raise ValidationError("Qdrant origin is not deployment-approved")


def probe_active_corpus(client: QdrantClient, snapshot: CorpusSnapshot, *, completed: bool) -> None:
    names = {item.name for item in client.get_collections().collections}
    physical = snapshot.physical_collection

    if physical not in names:
        if completed:
            raise ValidationError("Completed PostgreSQL chunks have no active Qdrant collection")

        return  # Fresh empty deployment; ingestion owns collection creation.

    info = client.get_collection(physical)
    sparse = info.config.params.sparse_vectors or {}

    if SPARSE_VECTOR_NAME not in sparse or sparse[SPARSE_VECTOR_NAME].modifier != Modifier.IDF:
        raise ValidationError("Active corpus has no compatible BM25 sparse index")

    if not info.points_count and physical + "__embedding_identity" not in names:
        if completed or getattr(info.config.params.vectors, "size", None) != settings.embed_dim:
            raise ValidationError("Active corpus has no compatible immutable identity")

        return

    validate_corpus_binding(
        client, snapshot.endpoint, physical, snapshot.embedding_identity, settings.embed_dim
    )
