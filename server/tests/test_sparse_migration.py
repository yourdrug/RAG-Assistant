"""Offline BM25 sparse migration keeps dense vectors, payload, aliases and ACL intact."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    Modifier,
    PointStruct,
    PointVectors,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from domain.shared.exceptions import ValidationError
from infrastructure.corpus.snapshot import CorpusSnapshot
from infrastructure.corpus.sparse_migration import (
    STAGING_SUFFIX,
    copy_points,
    create_target,
    migrate_sparse_vectors,
    verify_sparse_coverage,
)
from infrastructure.corpus.validation import validate_corpus
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME, encode_sparse


IDENTITY = "tei:model:revision"
PHYSICAL = "docs_v1"
ALIAS = "company_docs"
OLD_SPARSE = "bm25_ru_legacy"
SNAPSHOT = CorpusSnapshot("http://qdrant:6333", ALIAS, PHYSICAL, IDENTITY)
ROWS: list[dict[str, Any]] = [
    {
        "id": index,
        "document_id": 10 + index,
        "visibility": "internal_private" if index % 2 else "internal_public",
        "owner_id": 7 if index % 2 else None,
        "group_id": None,
        "content_hash": f"h{index}",
        "doc_domain": "general",
        "content": f"Статья {index}. Требования к договорам поставки",
    }
    for index in range(1, 6)
]


def legacy_corpus() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        PHYSICAL,
        vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        sparse_vectors_config={OLD_SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
    )
    client.create_collection(
        PHYSICAL + "__embedding_identity",
        vectors_config={
            hashlib.sha256(IDENTITY.encode()).hexdigest(): VectorParams(size=1, distance=Distance.COSINE)
        },
    )
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(create_alias=CreateAlias(collection_name=PHYSICAL, alias_name=ALIAS))
        ]
    )
    client.upsert(
        PHYSICAL,
        [
            PointStruct(
                id=row["id"],
                vector={"": [1.0, float(row["id"])], OLD_SPARSE: SparseVector(indices=[1], values=[1.0])},
                payload={
                    "page_content": row["content"],
                    "metadata": {
                        **{k: v for k, v in row.items() if k not in {"id", "content"}},
                        "embedding_model": IDENTITY,
                    },
                },
            )
            for row in ROWS
        ],
    )
    return client


def assert_migrated(client: QdrantClient) -> None:
    info = client.get_collection(PHYSICAL)
    assert set(info.config.params.sparse_vectors or {}) == {SPARSE_VECTOR_NAME}
    assert not client.collection_exists(PHYSICAL + STAGING_SUFFIX)

    points = {p.id: p for p in client.retrieve(PHYSICAL, [r["id"] for r in ROWS], with_vectors=True)}

    for row in ROWS:
        point = points[row["id"]]
        vectors = point.vector
        assert isinstance(vectors, dict)
        assert vectors[""] == pytest.approx(
            [v / (1 + row["id"] ** 2) ** 0.5 for v in (1.0, float(row["id"]))]
        )  # cosine collections store normalized dense vectors
        assert vectors[SPARSE_VECTOR_NAME] == encode_sparse(row["content"])
        assert point.payload is not None
        assert point.payload["metadata"]["owner_id"] == row["owner_id"]

    # The API's own startup/switch check passes through the restored alias.
    validate_corpus(client, SNAPSHOT.endpoint, ALIAS, IDENTITY, 2, rows=ROWS)


def test_migration_rebuilds_sparse_and_keeps_corpus() -> None:
    client = legacy_corpus()
    try:
        report = migrate_sparse_vectors(client, SNAPSHOT)

        assert (report.points, report.rebuilt) == (len(ROWS), True)
        assert_migrated(client)

        again = migrate_sparse_vectors(client, SNAPSHOT)
        assert (again.points, again.rebuilt) == (len(ROWS), False)
    finally:
        client.close()


def test_migration_resumes_after_original_was_dropped() -> None:
    client = legacy_corpus()
    try:
        staging = PHYSICAL + STAGING_SUFFIX
        create_target(client, staging, client.get_collection(PHYSICAL).config.params.vectors)
        copy_points(client, PHYSICAL, staging, reencode=True)
        client.delete_collection(PHYSICAL)  # alias disappears with the collection

        assert migrate_sparse_vectors(client, SNAPSHOT).rebuilt is True
        assert_migrated(client)
    finally:
        client.close()


def test_migration_resumes_interrupted_copy_back() -> None:
    client = legacy_corpus()
    try:
        staging = PHYSICAL + STAGING_SUFFIX
        dense = client.get_collection(PHYSICAL).config.params.vectors
        create_target(client, staging, dense)
        copy_points(client, PHYSICAL, staging, reencode=True)
        client.delete_collection(PHYSICAL)
        create_target(client, PHYSICAL, dense)  # recreated, nothing copied back yet

        assert migrate_sparse_vectors(client, SNAPSHOT).rebuilt is True
        assert_migrated(client)
    finally:
        client.close()


def test_incomplete_staging_is_rebuilt_from_intact_original() -> None:
    client = legacy_corpus()
    try:
        staging = PHYSICAL + STAGING_SUFFIX
        create_target(client, staging, client.get_collection(PHYSICAL).config.params.vectors)

        assert migrate_sparse_vectors(client, SNAPSHOT).rebuilt is True
        assert_migrated(client)
    finally:
        client.close()


def test_coverage_check_rejects_missing_and_empty_sparse_vectors() -> None:
    client = legacy_corpus()
    try:
        migrate_sparse_vectors(client, SNAPSHOT)
        ids = {row["id"] for row in ROWS}

        with pytest.raises(ValidationError, match="missing"):
            verify_sparse_coverage(client, PHYSICAL, ids | {99})

        client.update_vectors(
            PHYSICAL,
            points=[PointVectors(id=1, vector={SPARSE_VECTOR_NAME: SparseVector(indices=[], values=[])})],
        )
        with pytest.raises(ValidationError, match="no bm25"):
            verify_sparse_coverage(client, PHYSICAL, ids)
    finally:
        client.close()


def test_missing_collection_is_reported() -> None:
    client = QdrantClient(":memory:")
    try:
        with pytest.raises(ValidationError, match="does not exist"):
            migrate_sparse_vectors(client, SNAPSHOT)
    finally:
        client.close()
