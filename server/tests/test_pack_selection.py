"""A pack selection selects the same chunks in Qdrant as the reference semantics, under ACL."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qdrant_client import QdrantClient, models
from sqlalchemy.dialects import postgresql

from domain.identity.roles import UserKind, UserRole
from domain.knowledge.pack_metadata import PACK_METADATA_KEY, matches, pack_metadata_of
from infrastructure.repositories.chunk.pack_selection import pack_selection_clause
from infrastructure.repositories.vector.acl import build_qdrant_filter, build_search_filter
from tests.pack_selection_cases import PACK_METADATA, SELECTIONS


if TYPE_CHECKING:
    from collections.abc import Iterator

    from domain.knowledge.pack_metadata import MetadataCondition


COLLECTION = "pack_selection"
PRIVATE_ID = 100


@pytest.fixture(scope="module")
def client() -> Iterator[QdrantClient]:
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        COLLECTION, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
    )
    points = [
        models.PointStruct(
            id=index + 1,
            vector=[1.0, 0.0],
            payload={"metadata": {"visibility": "internal_public", PACK_METADATA_KEY: metadata}},
        )
        for index, metadata in enumerate(PACK_METADATA)
    ]
    # Matches every selection by its pack metadata but is private to another user.
    points.append(
        models.PointStruct(
            id=PRIVATE_ID,
            vector=[1.0, 0.0],
            payload={
                "metadata": {
                    "visibility": "internal_private",
                    "owner_id": 999,
                    PACK_METADATA_KEY: PACK_METADATA[2],
                }
            },
        )
    )
    qdrant.upsert(COLLECTION, points)
    yield qdrant
    qdrant.close()


USER = {"id": 5, "kind": UserKind.INTERNAL, "role": UserRole.USER}


def selected_ids(client: QdrantClient, selection: MetadataCondition | None) -> set[int]:
    points, _ = client.scroll(COLLECTION, scroll_filter=build_search_filter(USER, [], selection), limit=100)
    return {int(point.id) for point in points}


@pytest.mark.parametrize("name", sorted(SELECTIONS))
def test_qdrant_selection_matches_reference_semantics(client: QdrantClient, name: str) -> None:
    selection = SELECTIONS[name]
    expected = {index + 1 for index, metadata in enumerate(PACK_METADATA) if matches(selection, metadata)}
    assert selected_ids(client, selection) == expected


def test_selection_never_widens_access(client: QdrantClient) -> None:
    access_only, _ = client.scroll(COLLECTION, scroll_filter=build_qdrant_filter(USER, []), limit=100)
    assert PRIVATE_ID not in {int(point.id) for point in access_only}

    for selection in SELECTIONS.values():
        assert PRIVATE_ID not in selected_ids(client, selection)


def test_no_selection_keeps_the_access_filter_unchanged(client: QdrantClient) -> None:
    assert build_search_filter(USER, [], None) == build_qdrant_filter(USER, [])
    assert selected_ids(client, None) == set(range(1, len(PACK_METADATA) + 1))


@pytest.mark.parametrize("name", sorted(SELECTIONS))
def test_sql_selection_compiles_for_postgres(name: str) -> None:
    dialect = postgresql.dialect()  # type: ignore[no-untyped-call]  # SQLAlchemy exposes an untyped dialect constructor.
    compiled = str(pack_selection_clause(SELECTIONS[name]).compile(dialect=dialect))
    assert compiled


def test_pack_metadata_of_ignores_non_dictionaries() -> None:
    assert pack_metadata_of({PACK_METADATA_KEY: ["edition"]}) == {}
    assert pack_metadata_of({}) == {}
    assert pack_metadata_of({PACK_METADATA_KEY: {"edition": 1}}) == {"edition": 1}
