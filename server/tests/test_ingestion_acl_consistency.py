"""SEC-001: ingestion and direct chunk reads must agree on document ACL."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from application.chunks import ChunkQueryService
from application.ingestion import DocumentSyncService
from domain.identity.roles import UserKind, UserRole
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility, SourceType
from domain.knowledge.index_sync import OutboxOperation
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from application.chunks.dto import ChunkItemDTO
    from application.ingestion.dto import RegistryRecord
    from domain.knowledge.index_sync import VectorOutboxEntry


FILENAME = "report.md"
SOURCE = "s3://bucket/docs/report.md"
CLIENT_ID = 20


@pytest_asyncio.fixture
async def indexed_public_document() -> tuple[FakeUnitOfWorkFactory, int]:
    factory = FakeUnitOfWorkFactory()
    uow = factory._uow
    uow.fake_users.add_user(CLIENT_ID, kind=UserKind.CLIENT)
    doc = await uow.documents.save(Document(filename=FILENAME, ingestion_key="docs/report.md"))
    assert doc.id is not None
    await uow.chunks.bulk_insert(
        document_id=doc.id,
        filename=FILENAME,
        visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
        chunks=[ChunkInsert(content="old public text")],
    )
    return factory, doc.id


async def reindex_for_client(
    factory: FakeUnitOfWorkFactory,
    *,
    client_id: int = CLIENT_ID,
    registry: dict[str, RegistryRecord] | None = None,
) -> list[RawDocument]:
    async with factory.create(master=True) as uow:
        await uow.documents.set_acl(1, DocumentVisibility.CLIENT_PRIVATE, CLIENT_ID, None)

        for row in uow.fake_chunks._chunks:
            row.update(visibility=DocumentVisibility.CLIENT_PRIVATE.value, owner_id=CLIENT_ID)
    chunks = [
        RawDocument(
            page_content="new private text",
            metadata={
                "source": SOURCE,
                "visibility": DocumentVisibility.CLIENT_PRIVATE.value,
                "owner_id": None,  # Legacy ingestion tags client_id separately.
                "client_id": client_id,
            },
        )
    ]
    await DocumentSyncService(factory).sync_documents_to_db(
        registry or {"docs/report.md": {"source": SOURCE, "hash": "fixture", "chars": 0, "chunks": 0}},
        {},
        chunks,
        visibility=DocumentVisibility.CLIENT_PRIVATE,
        client_id=client_id,
    )
    return chunks


async def read_chunks(
    service: ChunkQueryService,
    document_id: int,
    *,
    cursor: bool,
    user_id: int,
    kind: UserKind,
    role: UserRole = UserRole.USER,
) -> list[ChunkItemDTO]:
    if cursor:
        page = await service.list_chunks_cursor(document_id, user_id, kind, role)
        return page.items

    items, total = await service.list_chunks(document_id, user_id, kind, role)
    assert total == len(items)
    return items


@pytest.mark.asyncio
async def test_reindex_persists_one_acl_for_document_chunks_and_outbox(
    indexed_public_document: tuple[FakeUnitOfWorkFactory, int],
) -> None:
    factory, doc_id = indexed_public_document
    produced_chunks = await reindex_for_client(factory)
    uow = factory._uow

    assert list(uow.fake_documents._documents) == [doc_id]
    doc = await uow.documents.get_by_id(doc_id)
    assert doc is not None
    assert doc.visibility == DocumentVisibility.CLIENT_PRIVATE
    assert doc.owner_id == CLIENT_ID
    assert doc.group_id is None
    rows, total = await uow.chunks.list_for_document(doc_id)
    assert total == 1
    assert rows[0].content == "new private text"
    assert rows[0].visibility == DocumentVisibility.CLIENT_PRIVATE.value
    assert rows[0].owner_id == CLIENT_ID
    assert rows[0].group_id is None
    assert produced_chunks[0].metadata["client_id"] == CLIENT_ID
    upserts = [
        entry
        for entry in uow.fake_vector_outbox._entries.values()
        if entry.operation == OutboxOperation.UPSERT_CHUNKS
    ]
    metadata = upserts[0].payload["points"][0]["metadata"]
    assert metadata["visibility"] == DocumentVisibility.CLIENT_PRIVATE.value
    assert metadata["owner_id"] == CLIENT_ID
    assert metadata["group_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [False, True], ids=["offset", "cursor"])
async def test_reindexed_private_content_allows_client_and_denies_unrelated_users(
    indexed_public_document: tuple[FakeUnitOfWorkFactory, int], cursor: bool
) -> None:
    factory, doc_id = indexed_public_document
    await reindex_for_client(factory)
    service = ChunkQueryService(factory)

    items = await read_chunks(service, doc_id, cursor=cursor, user_id=CLIENT_ID, kind=UserKind.CLIENT)
    assert [item.content for item in items] == ["new private text"]

    for user_id, kind in [(10, UserKind.INTERNAL), (21, UserKind.CLIENT)]:
        with pytest.raises(BusinessRuleViolationError):
            await read_chunks(service, doc_id, cursor=cursor, user_id=user_id, kind=kind)


@pytest.mark.asyncio
async def test_repeated_client_ingestion_reuses_the_owned_document(
    indexed_public_document: tuple[FakeUnitOfWorkFactory, int],
) -> None:
    factory, doc_id = indexed_public_document
    await reindex_for_client(factory)
    await reindex_for_client(factory)

    assert list(factory._uow.fake_documents._documents) == [doc_id]
    rows, total = await factory._uow.chunks.list_for_document(doc_id)
    assert total == 1
    assert rows[0].owner_id == CLIENT_ID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_type,ingestion_key,source_path",
    [
        (SourceType.FILE, None, "s3://bucket/other/report.md"),
        (SourceType.FILE, "other/report.md", "s3://bucket/other/report.md"),
        (SourceType.FILE, None, "uploads/report.md"),
        (SourceType.FILE, None, ""),
        (SourceType.MANUAL, None, SOURCE),
    ],
    ids=["legacy-other-source", "other-ingestion-key", "upload", "unknown-source", "manual"],
)
async def test_private_import_does_not_claim_same_named_public_document(
    source_type: SourceType, ingestion_key: str | None, source_path: str
) -> None:
    factory = FakeUnitOfWorkFactory()
    uow = factory._uow
    uow.fake_users.add_user(CLIENT_ID, kind=UserKind.CLIENT)
    original = await uow.documents.save(
        Document(
            filename=FILENAME,
            source_type=source_type.value,
            ingestion_key=ingestion_key,
            source_path=source_path,
        )
    )
    assert original.id is not None
    await uow.chunks.bulk_insert(
        original.id,
        FILENAME,
        DocumentVisibility.INTERNAL_PUBLIC.value,
        [ChunkInsert(content="old public text")],
    )
    await DocumentSyncService(factory).sync_documents_to_db(
        {"docs/report.md": {"source": SOURCE, "hash": "fixture", "chars": 0, "chunks": 0}},
        {},
        [
            RawDocument(
                page_content="private text",
                metadata={"source": SOURCE, "hash": "fixture", "chars": 0, "chunks": 0},
            )
        ],
        visibility=DocumentVisibility.CLIENT_PRIVATE,
        client_id=CLIENT_ID,
    )

    public = await uow.documents.get_by_id(original.id)

    assert public is not None
    assert public.visibility == DocumentVisibility.INTERNAL_PUBLIC
    assert public.owner_id is None
    assert public.source_path == source_path
    rows, total = await uow.chunks.list_for_document(original.id)
    assert total == 1
    assert rows[0].content == "old public text"
    (imported,) = await uow.documents.find_ingestion_documents(["docs/report.md"])
    assert imported.id != original.id
    assert imported.visibility == DocumentVisibility.CLIENT_PRIVATE
    assert imported.owner_id == CLIENT_ID
    assert imported.source_path == "docs/report.md"


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_outbox", [False, True], ids=["commit", "rollback"])
async def test_reindex_updates_source_path_atomically_with_chunks(
    indexed_public_document: tuple[FakeUnitOfWorkFactory, int],
    monkeypatch: pytest.MonkeyPatch,
    fail_outbox: bool,
) -> None:
    factory, doc_id = indexed_public_document
    uow = factory._uow
    old_source = "docs/old-report.md"
    await uow.documents.set_source_path(doc_id, old_source)

    if fail_outbox:

        async def fail_enqueue(entry: VectorOutboxEntry) -> None:
            message = "outbox unavailable"
            raise RuntimeError(message)

        monkeypatch.setattr(uow.vector_outbox, "enqueue", fail_enqueue)
        with pytest.raises(RuntimeError, match="outbox unavailable"):
            await reindex_for_client(factory)
    else:
        await reindex_for_client(factory)

    document = await uow.documents.get_by_id(doc_id)
    assert document is not None
    assert document.source_path == (old_source if fail_outbox else "docs/report.md")
    rows, total = await uow.chunks.list_for_document(doc_id)
    assert total == 1
    assert rows[0].content == ("old public text" if fail_outbox else "new private text")


@pytest.mark.asyncio
@pytest.mark.parametrize("client_id", [None, 999, 30], ids=["missing", "unknown", "internal-user"])
async def test_invalid_client_scope_is_rejected_before_replacing_content(
    indexed_public_document: tuple[FakeUnitOfWorkFactory, int], client_id: int | None
) -> None:
    factory, doc_id = indexed_public_document
    factory._uow.fake_users.add_user(30, kind=UserKind.INTERNAL)

    chunks = [
        RawDocument(
            page_content="new",
            metadata={"source": "s3://bucket/docs/new.md", "hash": "fixture", "chars": 0, "chunks": 0},
        )
    ]
    with pytest.raises(ValidationError):
        await DocumentSyncService(factory).sync_documents_to_db(
            {
                "docs/new.md": {
                    "source": "s3://bucket/docs/new.md",
                    "hash": "fixture",
                    "chars": 0,
                    "chunks": 0,
                }
            },
            {},
            chunks,
            visibility=DocumentVisibility.CLIENT_PRIVATE,
            client_id=client_id,
        )

    doc = await factory._uow.documents.get_by_id(doc_id)

    assert doc is not None
    assert doc.visibility == DocumentVisibility.INTERNAL_PUBLIC
    assert doc.owner_id is None
    rows, total = await factory._uow.chunks.list_for_document(doc_id)
    assert total == 1
    assert rows[0].content == "old public text"
    assert factory._uow.fake_vector_outbox._entries == {}


@pytest.mark.asyncio
async def test_outbox_failure_rolls_back_acl_and_chunk_replacement(
    indexed_public_document: tuple[FakeUnitOfWorkFactory, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    factory, doc_id = indexed_public_document
    uow = factory._uow
    enqueue = uow.vector_outbox.enqueue

    async def fail_after_upsert(entry: VectorOutboxEntry) -> None:
        if entry.operation == OutboxOperation.UPSERT_CHUNKS:
            message = "outbox unavailable"
            raise RuntimeError(message)

        await enqueue(entry)

    monkeypatch.setattr(uow.vector_outbox, "enqueue", fail_after_upsert)
    with pytest.raises(RuntimeError, match="outbox unavailable"):
        await reindex_for_client(factory)

    doc = await uow.documents.get_by_id(doc_id)

    assert doc is not None
    assert doc.visibility == DocumentVisibility.CLIENT_PRIVATE
    assert doc.owner_id == CLIENT_ID
    rows, total = await uow.chunks.list_for_document(doc_id)
    assert total == 1
    assert rows[0].content == "old public text"
    assert rows[0].visibility == DocumentVisibility.CLIENT_PRIVATE.value
    assert uow.fake_vector_outbox._entries == {}


@pytest.mark.asyncio
async def test_registry_entries_without_new_chunks_keep_their_acl(
    indexed_public_document: tuple[FakeUnitOfWorkFactory, int],
) -> None:
    factory, _ = indexed_public_document
    other = await factory._uow.documents.save(Document(filename="other.md"))
    assert other.id is not None
    registry: dict[str, RegistryRecord] = {
        "docs/report.md": {"source": SOURCE, "hash": "fixture", "chars": 0, "chunks": 0},
        "other.md": {"source": "s3://bucket/docs/other.md", "hash": "fixture", "chars": 0, "chunks": 0},
    }
    await reindex_for_client(factory, registry=registry)

    stored = await factory._uow.documents.get_by_id(other.id)

    assert stored is not None
    assert stored.visibility == DocumentVisibility.INTERNAL_PUBLIC
    assert stored.owner_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [False, True], ids=["offset", "cursor"])
@pytest.mark.parametrize(
    "chunk_acl",
    [
        {"visibility": DocumentVisibility.CLIENT_PRIVATE.value, "owner_id": CLIENT_ID},
        {"owner_id": CLIENT_ID},
        {"group_id": 7},
    ],
    ids=["visibility", "owner", "group"],
)
async def test_legacy_chunk_acl_mismatch_blocks_listing(
    indexed_public_document: tuple[FakeUnitOfWorkFactory, int],
    cursor: bool,
    chunk_acl: dict[str, object] | dict[str, int],
) -> None:
    factory, doc_id = indexed_public_document
    factory._uow.fake_chunks._chunks[0].update(chunk_acl)

    with pytest.raises(BusinessRuleViolationError, match="ACL"):
        await read_chunks(
            ChunkQueryService(factory), doc_id, cursor=cursor, user_id=10, kind=UserKind.INTERNAL
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [False, True], ids=["offset", "cursor"])
async def test_stale_public_document_snapshot_cannot_authorize_private_chunks(
    indexed_public_document: tuple[FakeUnitOfWorkFactory, int], cursor: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory, doc_id = indexed_public_document
    public_snapshot = deepcopy(await factory._uow.documents.get_by_id(doc_id))
    await reindex_for_client(factory)

    async def read_snapshot(document_id: int) -> Document | None:
        # Simulate the document read occurring before the concurrent commit,
        # with the subsequent chunk read observing the new committed content.
        return deepcopy(public_snapshot)

    monkeypatch.setattr(factory._uow.documents, "get_by_id", read_snapshot)
    service = ChunkQueryService(factory)
    with pytest.raises(BusinessRuleViolationError, match="ACL"):
        await read_chunks(service, doc_id, cursor=cursor, user_id=10, kind=UserKind.INTERNAL)
