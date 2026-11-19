"""Offline rebuild of BM25 sparse vectors after a ``SPARSE_VECTOR_NAME`` bump.

Qdrant cannot add a named vector to an existing collection, so the active physical collection
is rebuilt under the same name: points are copied to a staging collection with sparse vectors
re-encoded from ``page_content`` (dense vectors, ids and payload unchanged), the collection is
recreated with the current sparse config and the points are copied back. Names, aliases and the
embedding identity binding stay the same, so the corpus journal remains valid.

Every step is restartable: the original collection is deleted only after the staging copy has
been verified point by point, and a run that finds staging next to a current collection
finishes the copy-back.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any

from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
)

from domain.shared.exceptions import ValidationError
from infrastructure.bm25.tokenizer import tokenize
from infrastructure.corpus.snapshot import BoundQdrantClient
from infrastructure.repositories.vector.qdrant_ops import ensure_payload_indexes
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME, encode_sparse


if TYPE_CHECKING:
    from collections.abc import Iterator

    from qdrant_client import QdrantClient
    from qdrant_client.models import CollectionInfo, Record

    from infrastructure.corpus.snapshot import CorpusSnapshot


log = logging.getLogger("cli")

STAGING_SUFFIX = "__sparse_staging"
SCROLL_BATCH = 256


@dataclass(frozen=True)
class SparseMigrationReport:
    collection: str
    points: int
    rebuilt: bool


def has_current_sparse(info: CollectionInfo) -> bool:
    sparse = info.config.params.sparse_vectors or {}
    return SPARSE_VECTOR_NAME in sparse and sparse[SPARSE_VECTOR_NAME].modifier == Modifier.IDF


def scroll_points(
    client: QdrantClient, collection: str, *, with_vectors: bool | list[str]
) -> Iterator[Record]:
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=SCROLL_BATCH,
            with_payload=True,
            with_vectors=with_vectors,
        )
        yield from points

        if offset is None:
            return


def dense_vector(point: Record) -> Any:  # noqa: ANN401 — Qdrant returns list[float] or multivector.
    vectors = point.vector

    dense = vectors.get("") if isinstance(vectors, dict) else vectors

    if dense is None:
        message = f"Point {point.id} has no dense vector; re-ingest it instead of migrating"
        raise ValidationError(message)

    return dense


def copy_points(client: QdrantClient, source: str, target: str, *, reencode: bool) -> int:
    """Copy all points; ``reencode`` rebuilds the sparse vector from ``page_content``."""
    batch: list[PointStruct] = []
    copied = 0

    for point in scroll_points(client, source, with_vectors=True):
        payload = point.payload or {}

        if reencode:
            sparse = encode_sparse(str(payload.get("page_content") or ""))
        else:
            vectors = point.vector if isinstance(point.vector, dict) else {}
            stored = vectors.get(SPARSE_VECTOR_NAME)

            if not isinstance(stored, SparseVector):
                message = f"Staging point {point.id} has no {SPARSE_VECTOR_NAME} vector"
                raise ValidationError(message)

            sparse = stored

        batch.append(
            PointStruct(
                id=point.id, vector={"": dense_vector(point), SPARSE_VECTOR_NAME: sparse}, payload=payload
            )
        )

        if len(batch) >= SCROLL_BATCH:
            client.upsert(collection_name=target, points=batch, wait=True)
            copied += len(batch)
            batch = []

    if batch:
        client.upsert(collection_name=target, points=batch, wait=True)
        copied += len(batch)

    return copied


def point_ids(client: QdrantClient, collection: str) -> set[int | str]:
    return {point.id for point in scroll_points(client, collection, with_vectors=False)}


def verify_sparse_coverage(client: QdrantClient, collection: str, expected_ids: set[int | str]) -> int:
    """Every expected point is present once and carries a current, non-empty-when-tokenized sparse vector."""
    seen: set[int | str] = set()

    for point in scroll_points(client, collection, with_vectors=[SPARSE_VECTOR_NAME]):
        if point.id not in expected_ids or point.id in seen:
            message = f"Collection {collection} has an unexpected or duplicate point {point.id}"
            raise ValidationError(message)

        seen.add(point.id)
        vectors = point.vector if isinstance(point.vector, dict) else {}
        sparse = vectors.get(SPARSE_VECTOR_NAME)
        content = str((point.payload or {}).get("page_content") or "")

        if not isinstance(sparse, SparseVector) or (not sparse.indices and tokenize(content)):
            message = f"Point {point.id} in {collection} has no {SPARSE_VECTOR_NAME} vector"
            raise ValidationError(message)

    if missing := expected_ids - seen:
        message = (
            f"Collection {collection} is missing {len(missing)} points, e.g. {sorted(map(str, missing))[:5]}"
        )
        raise ValidationError(message)

    return len(seen)


def create_target(client: QdrantClient, name: str, dense_config: Any) -> None:  # noqa: ANN401 — vendor config.
    client.create_collection(
        collection_name=name,
        vectors_config=dense_config,
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )


def build_staging(client: QdrantClient, physical: str, info: CollectionInfo, staging: str) -> set[int | str]:
    if client.collection_exists(staging):
        log.info("Dropping incomplete staging collection %s", staging)
        client.delete_collection(staging)

    create_target(client, staging, info.config.params.vectors)
    copied = copy_points(client, physical, staging, reencode=True)
    expected = point_ids(client, physical)
    verify_sparse_coverage(client, staging, expected)
    log.info("Staged %d points with %s vectors in %s", copied, SPARSE_VECTOR_NAME, staging)
    return expected


def restore_alias(client: QdrantClient, alias: str, physical: str) -> None:
    if alias == physical:
        return

    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}

    if aliases.get(alias) != physical:
        client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(create_alias=CreateAlias(collection_name=physical, alias_name=alias))
            ]
        )


def migrate_sparse_vectors(client: QdrantClient, snapshot: CorpusSnapshot) -> SparseMigrationReport:
    """Bring ``snapshot.physical_collection`` to the current sparse encoder without re-embedding."""
    physical = snapshot.physical_collection
    staging = physical + STAGING_SUFFIX
    physical_exists = client.collection_exists(physical)
    staging_exists = client.collection_exists(staging)

    if physical_exists:
        info = client.get_collection(physical)

        if has_current_sparse(info) and not staging_exists:
            expected = point_ids(client, physical)
            points = verify_sparse_coverage(client, physical, expected)
            return SparseMigrationReport(physical, points, rebuilt=False)

        if has_current_sparse(info):
            # Interrupted copy-back: staging is complete, the collection was already recreated.
            expected = point_ids(client, staging)
        else:
            expected = build_staging(client, physical, info, staging)
            log.info("Recreating %s with the %s sparse index", physical, SPARSE_VECTOR_NAME)
            client.delete_collection(physical)
            create_target(client, physical, info.config.params.vectors)
    elif staging_exists:
        # Interrupted after the original was dropped: staging was verified before that.
        expected = point_ids(client, staging)
        create_target(client, physical, client.get_collection(staging).config.params.vectors)
    else:
        message = f"Collection {physical} does not exist"
        raise ValidationError(message)

    copy_points(client, staging, physical, reencode=False)
    ensure_payload_indexes(BoundQdrantClient(client, snapshot))
    restore_alias(client, snapshot.alias, physical)
    points = verify_sparse_coverage(client, physical, expected)
    client.delete_collection(staging)
    log.info("Migrated %d points of %s to %s", points, physical, SPARSE_VECTOR_NAME)
    return SparseMigrationReport(physical, points, rebuilt=True)
