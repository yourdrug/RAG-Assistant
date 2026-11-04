"""Read-only validation: a candidate must match the PostgreSQL corpus and ACL."""

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Modifier, Record, SparseVector

from domain.shared.exceptions import ValidationError
from infrastructure.bm25.tokenizer import tokenize
from infrastructure.corpus.snapshot import CorpusSnapshot
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME


ACL_FIELDS = ("document_id", "visibility", "owner_id", "group_id", "content_hash", "doc_domain")


def corpus_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        repr(sorted((row["id"], *(row[key] for key in ACL_FIELDS), row["content"]) for row in rows)).encode()
    ).hexdigest()


def validate_corpus(
    client: QdrantClient,
    endpoint: str,
    name: str,
    identity: str,
    dimension: int,
    *,
    rows: Sequence[Mapping[str, Any]],
) -> CorpusSnapshot:
    snapshot = validate_corpus_binding(client, endpoint, name, identity, dimension)
    physical = snapshot.physical_collection
    expected = {row["id"]: row for row in rows}
    seen: set[int | str] = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=physical,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=[SPARSE_VECTOR_NAME],
        )

        for point in points:
            validate_corpus_point(point, expected, seen, identity)
            seen.add(point.id)

        if offset is None:
            break

    if seen != set(expected):
        message = "Selected corpus is missing PostgreSQL chunks"
        raise ValidationError(message)

    return CorpusSnapshot(endpoint, name, physical, identity)


def validate_corpus_point(
    point: Record, expected: Mapping[int | str, Mapping[str, Any]], seen: set[int | str], identity: str
) -> None:
    row = expected.get(point.id)
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}

    if row is None or point.id in seen:
        message = "Selected corpus contains unknown or duplicate chunk IDs"
        raise ValidationError(message)

    if any(key not in metadata or metadata[key] != row[key] for key in ACL_FIELDS):
        message = "Selected corpus ACL/content metadata differs from PostgreSQL"
        raise ValidationError(message)

    if metadata.get("embedding_model") != identity or payload.get("page_content") != row["content"]:
        message = "Selected corpus content or embedding identity differs from PostgreSQL"
        raise ValidationError(message)

    vectors = point.vector
    sparse = vectors.get(SPARSE_VECTOR_NAME) if isinstance(vectors, dict) else None

    if not isinstance(sparse, SparseVector):
        message = "Selected corpus chunk has no BM25 sparse vector"
        raise ValidationError(message)

    if not sparse.indices and tokenize(row["content"]):
        message = "Selected corpus chunk has an empty BM25 sparse vector for tokenized content"
        raise ValidationError(message)


def validate_corpus_binding(
    client: QdrantClient, endpoint: str, name: str, identity: str, dimension: int
) -> CorpusSnapshot:
    names = {item.name for item in client.get_collections().collections}
    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    physical = aliases.get(name, name)

    if physical not in names:
        message = "Selected corpus does not exist"
        raise ValidationError(message)

    info = client.get_collection(physical)
    vectors = info.config.params.vectors

    if getattr(vectors, "size", None) != dimension:
        message = "Selected corpus has incompatible vector dimension"
        raise ValidationError(message)

    sparse = info.config.params.sparse_vectors or {}

    if SPARSE_VECTOR_NAME not in sparse or sparse[SPARSE_VECTOR_NAME].modifier != Modifier.IDF:
        message = "Selected corpus has no compatible BM25 sparse index"
        raise ValidationError(message)

    binding = physical + "__embedding_identity"
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()

    if binding not in names:
        message = "Selected corpus has no immutable embedding identity"
        raise ValidationError(message)

    binding_vectors = client.get_collection(binding).config.params.vectors

    if not isinstance(binding_vectors, dict) or set(binding_vectors) != {fingerprint}:
        message = "Selected corpus has incompatible embedding identity"
        raise ValidationError(message)

    return CorpusSnapshot(endpoint, name, physical, identity)
