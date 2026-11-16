"""Admin chunk inspector: Postgres chunk versus its vector-index point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qdrant_client.models import SparseVector

from application.chunks import ChunkIndexInspectionService, IndexedPointSnapshot
from domain.identity.capabilities import Capability, get_role_capabilities
from domain.identity.roles import UserKind, UserRole
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility
from domain.knowledge.repositories.chunk import ChunkInsertOptions
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError
from infrastructure.repositories.vector.qdrant_vector_store_repository import snapshot_from_record
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from application.chunks import ChunkIndexStateDTO


ADMIN_ID = 1
OWNER_ID = 15
GROUP_ID = 7
SPACE_ID = 1  # the fake repository assigns the default space to new documents
CONTENT = "3.2. Срок рассмотрения заявки составляет не более 10 рабочих дней."


class FakeInspector:
    def __init__(self) -> None:
        self.points: dict[int, IndexedPointSnapshot] = {}
        self.calls: list[int] = []

    async def inspect_point(self, point_id: int) -> IndexedPointSnapshot | None:
        self.calls.append(point_id)
        return self.points.get(point_id)


async def seed() -> tuple[FakeUnitOfWorkFactory, int, int]:
    """Seed one indexed-group document; its default knowledge space is SPACE_ID."""
    factory = FakeUnitOfWorkFactory()
    uow = factory._uow
    uow.fake_users.add_user(ADMIN_ID)
    doc = await uow.documents.save(
        Document(
            filename="reglament.pdf",
            visibility=DocumentVisibility.INTERNAL_GROUP,
            owner_id=OWNER_ID,
            group_id=GROUP_ID,
        )
    )
    assert doc.id is not None
    assert doc.space_id == SPACE_ID
    [chunk_id] = await uow.chunks.bulk_insert(
        document_id=doc.id,
        filename="reglament.pdf",
        visibility=DocumentVisibility.INTERNAL_GROUP.value,
        chunks=[ChunkInsert(content=CONTENT)],
        options=ChunkInsertOptions(
            owner_id=OWNER_ID, group_id=GROUP_ID, space_id=doc.space_id, doc_domain="general"
        ),
    )
    return factory, doc.id, chunk_id


def matching_point(document_id: int, chunk_id: int, **overrides: object) -> IndexedPointSnapshot:
    metadata: dict[str, object] = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "visibility": DocumentVisibility.INTERNAL_GROUP.value,
        "owner_id": OWNER_ID,
        "group_id": GROUP_ID,
        "space_id": SPACE_ID,
        "doc_domain": "general",
        "embedding_model": "bge-m3",
    }
    metadata.update(overrides)
    return IndexedPointSnapshot(metadata=metadata, page_content=CONTENT, dense_dim=1024, sparse_terms=9)


async def inspect_as(
    service: ChunkIndexInspectionService,
    document_id: int,
    chunk_id: int,
    role: UserRole = UserRole.ADMIN,
    kind: UserKind = UserKind.INTERNAL,
) -> ChunkIndexStateDTO:
    return await service.inspect(document_id, chunk_id, ADMIN_ID, kind, role)


@pytest.mark.asyncio
async def test_admin_sees_point_in_sync_with_postgres() -> None:
    factory, doc_id, chunk_id = await seed()
    inspector = FakeInspector()
    inspector.points[chunk_id] = matching_point(doc_id, chunk_id)

    state = await inspect_as(ChunkIndexInspectionService(factory, inspector), doc_id, chunk_id)

    assert state.indexed is True
    assert state.in_sync is True
    assert state.content_matches is True
    assert state.dense_dim == 1024
    assert state.sparse_terms == 9
    assert state.embedding_model == "bge-m3"
    assert all(f.matches for f in state.fields)
    assert {f.name for f in state.fields} >= {"visibility", "owner_id", "group_id", "document_id"}


@pytest.mark.asyncio
async def test_acl_drift_in_point_is_reported() -> None:
    factory, doc_id, chunk_id = await seed()
    inspector = FakeInspector()
    inspector.points[chunk_id] = matching_point(doc_id, chunk_id, group_id=4)

    state = await inspect_as(ChunkIndexInspectionService(factory, inspector), doc_id, chunk_id)

    assert state.in_sync is False
    drifted = [f for f in state.fields if not f.matches]
    assert [(f.name, f.expected, f.actual) for f in drifted] == [("group_id", GROUP_ID, 4)]


@pytest.mark.asyncio
async def test_stale_content_and_missing_sparse_vector_break_sync() -> None:
    factory, doc_id, chunk_id = await seed()
    inspector = FakeInspector()
    point = matching_point(doc_id, chunk_id)
    inspector.points[chunk_id] = IndexedPointSnapshot(
        metadata=point.metadata, page_content="old text", dense_dim=1024, sparse_terms=None
    )

    state = await inspect_as(ChunkIndexInspectionService(factory, inspector), doc_id, chunk_id)

    assert state.content_matches is False
    assert state.in_sync is False


@pytest.mark.asyncio
async def test_missing_point_is_reported_as_not_indexed() -> None:
    factory, doc_id, chunk_id = await seed()

    state = await inspect_as(ChunkIndexInspectionService(factory, FakeInspector()), doc_id, chunk_id)

    assert state.indexed is False
    assert state.in_sync is False
    assert state.fields == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role, kind",
    [
        (UserRole.CURATOR, UserKind.INTERNAL),
        (UserRole.USER, UserKind.INTERNAL),
        (UserRole.ADMIN, UserKind.CLIENT),
    ],
)
async def test_non_admin_cannot_inspect_and_index_is_not_read(role: UserRole, kind: UserKind) -> None:
    factory, doc_id, chunk_id = await seed()
    inspector = FakeInspector()
    inspector.points[chunk_id] = matching_point(doc_id, chunk_id)

    with pytest.raises(BusinessRuleViolationError):
        await inspect_as(ChunkIndexInspectionService(factory, inspector), doc_id, chunk_id, role, kind)

    assert inspector.calls == []


@pytest.mark.asyncio
async def test_chunk_of_another_document_is_not_found() -> None:
    factory, _doc_id, chunk_id = await seed()
    other = await factory._uow.documents.save(Document(filename="other.pdf"))
    assert other.id is not None
    inspector = FakeInspector()

    with pytest.raises(EntityNotFoundError):
        await inspect_as(ChunkIndexInspectionService(factory, inspector), other.id, chunk_id)

    assert inspector.calls == []


def test_inspect_capability_is_admin_only() -> None:
    assert Capability.CHUNKS_INSPECT_INDEX in get_role_capabilities(UserRole.ADMIN)
    assert Capability.CHUNKS_INSPECT_INDEX not in get_role_capabilities(UserRole.CURATOR)
    assert Capability.CHUNKS_INSPECT_INDEX not in get_role_capabilities(UserRole.USER)


def test_snapshot_exposes_vector_shapes_not_values() -> None:
    snapshot = snapshot_from_record(
        {"page_content": CONTENT, "metadata": {"group_id": GROUP_ID}},
        {"": [0.1] * 8, SPARSE_VECTOR_NAME: SparseVector(indices=[1, 5, 9], values=[0.2, 0.3, 0.4])},
    )

    assert snapshot.dense_dim == 8
    assert snapshot.sparse_terms == 3
    assert snapshot.metadata == {"group_id": GROUP_ID}
    assert snapshot.page_content == CONTENT
