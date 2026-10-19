"""Batch reads preserve query results, ACL scope and constant round-trip counts."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest

from application.act_versioning import ActVersionQueries
from application.documents import DocumentQueryService
from application.ingestion import (
    DocumentSyncService,
    IngestionRegistry,
    IngestionService,
    S3IngestionTargets,
    SingleFileIngestionWorkflow,
)
from application.ingestion.registry import s3_file_hash
from application.ports.file_storage import FileItem
from domain.entities.document import Document
from domain.entities.raw_document import RawDocument
from domain.entities.vector_outbox_entry import OutboxStatus, VectorOutboxEntry
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.visibility import DocumentVisibility


if TYPE_CHECKING:
    from typing import Any

    from tests.fakes import FakeUnitOfWorkFactory

    from application.ingestion.dto import RegistryRecord
else:
    from fakes import FakeUnitOfWorkFactory


@pytest.mark.asyncio
async def test_filenames_use_one_batch_and_ignore_missing_and_duplicate_ids() -> None:
    factory: Any = FakeUnitOfWorkFactory()
    documents = factory._uow.documents
    doc = await documents.save(Document(filename="report.pdf"))
    documents.get_by_ids = AsyncMock(wraps=documents.get_by_ids)
    documents.get_by_id = AsyncMock(side_effect=AssertionError("per-document read"))
    queries = ActVersionQueries(factory)
    assert await queries.get_document_filenames([doc.id, 999, doc.id]) == {doc.id: "report.pdf"}
    documents.get_by_ids.assert_awaited_once_with([doc.id, 999])
    assert await queries.get_document_filenames([]) == {}
    assert documents.get_by_ids.await_count == 1


@pytest.mark.asyncio
async def test_document_listing_batches_outbox_and_keeps_details_and_zero_counts() -> None:
    factory: Any = FakeUnitOfWorkFactory()
    uow = factory._uow
    docs = [
        await uow.documents.save(Document(filename=f"{i}.pdf", status=status))
        for i, status in enumerate([DocumentStatus.INDEXING] * 3 + [DocumentStatus.DONE])
    ]

    for doc, status in [
        (docs[0], OutboxStatus.PENDING),
        (docs[1], OutboxStatus.FAILED),
        (docs[1], OutboxStatus.DEAD_LETTER),
        (docs[3], OutboxStatus.FAILED),
    ]:
        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(aggregate_id=doc.id, status=status, last_error="error")
        )

    outbox = uow.vector_outbox
    outbox.count_by_documents = AsyncMock(wraps=outbox.count_by_documents)
    outbox.get_failed_details_by_documents = AsyncMock(wraps=outbox.get_failed_details_by_documents)
    service = DocumentQueryService(factory)
    result = await service.list_documents(1, "internal", "admin")
    assert [(dto.outbox_pending, dto.outbox_failed) for dto in result] == [(1, 0), (1, 2), (0, 0), (0, 0)]
    assert result[0].outbox_failed_details is None
    assert result[1].outbox_failed_details is not None
    assert len(result[1].outbox_failed_details) == 2
    assert result[2].outbox_failed_details is None
    assert result[3].outbox_failed_details is None
    outbox.count_by_documents.assert_awaited_once_with([doc.id for doc in docs[:3]])
    outbox.get_failed_details_by_documents.assert_awaited_once_with([docs[1].id])

    for doc in docs:
        doc.status = DocumentStatus.DONE

    await service.list_documents(1, "internal", "admin")
    assert outbox.count_by_documents.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_id,group_id", [(None, None), (None, 7), (20, None)])
async def test_ingestion_prefetches_slots_with_scope_and_legacy_fallback(
    owner_id: int | None, group_id: int | None
) -> None:
    factory: Any = FakeUnitOfWorkFactory()
    uow = factory._uow
    uow.users.add_user(20, kind="client")
    own = await uow.documents.save(Document(filename="own.pdf", owner_id=owner_id, group_id=group_id))
    legacy = await uow.documents.save(Document(filename="legacy.pdf", group_id=group_id))
    await uow.documents.save(Document(filename="own.pdf", owner_id=99, group_id=group_id))
    await uow.documents.save(Document(filename="new.pdf", owner_id=99, group_id=group_id))
    await uow.documents.save(Document(filename="new.pdf", group_id=99))
    uow.documents.find_active_slots_by_filenames = AsyncMock(
        wraps=uow.documents.find_active_slots_by_filenames
    )
    uow.documents.find_active_slot = AsyncMock(side_effect=AssertionError("per-document read"))
    visibility = (
        DocumentVisibility.CLIENT_PRIVATE
        if owner_id
        else (DocumentVisibility.INTERNAL_GROUP if group_id else DocumentVisibility.INTERNAL_PUBLIC)
    )
    registry: dict[str, RegistryRecord] = {
        name: {"hash": "hash", "source": name, "chunks": 1, "chars": 10}
        for name in ["own.pdf", "legacy.pdf", "new.pdf"]
    }
    result = await DocumentSyncService(factory).ensure_document_rows(registry, visibility, group_id, owner_id)
    assert result["own.pdf"] == own.id
    assert result["legacy.pdf"] == legacy.id
    created = await uow.documents.get_by_id(result["new.pdf"])
    assert (created.owner_id, created.group_id, created.visibility) == (owner_id, group_id, visibility)
    calls = uow.documents.find_active_slots_by_filenames.await_args_list
    assert len(calls) == (2 if owner_id else 1)
    assert calls[0].kwargs == {"owner_id": owner_id, "group_id": group_id}

    if owner_id:
        assert calls[1].args == (["legacy.pdf", "new.pdf"],)
        assert calls[1].kwargs == {"owner_id": None, "group_id": group_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("force", [False, True])
async def test_run_single_force_reprocesses_an_unchanged_registered_file(force: bool) -> None:
    factory: Any = FakeUnitOfWorkFactory()
    file = FileItem("docs/report.pdf", "report.pdf", 100, "2026-01-01", ".pdf")
    storage = Mock(supported_extensions=(".pdf",))
    storage.get_file_info = AsyncMock(return_value=file)
    registry = IngestionRegistry(factory, storage)
    await registry.upsert(file.filename, s3_file_hash(file), file.key, 1, chars=3)
    loader = Mock()
    loader.load_file = AsyncMock(return_value=[RawDocument(page_content="new text")])
    loader.index_docs.return_value = [RawDocument(page_content="new text")]
    loader.s3_source_key.return_value = file.key
    loader.classify_text_domain.return_value = "general"
    sync = AsyncMock()
    service = IngestionService(
        Mock(),
        SingleFileIngestionWorkflow(storage, loader, registry, sync),
        registry,
        S3IngestionTargets(),
        Mock(),
    )

    result = await service.run_single(file.key, force=force)

    assert result.force is force
    assert loader.load_file.await_count == int(force)
    assert sync.sync_documents_to_db.await_count == int(force)
    assert await registry.is_indexed(file.filename, s3_file_hash(file))
