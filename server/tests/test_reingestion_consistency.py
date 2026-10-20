"""Replacement must converge SQL, vector outbox and the sparse corpus."""

from types import SimpleNamespace
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from application.indexing import process_chunks
from application.ingestion import SingleFileIngestionWorkflow
from application.ingestion import IngestionScope
from domain.knowledge.chunk import ChunkInsert
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.index_sync import OutboxOperation, OutboxStatus
from domain.knowledge.enums import DocDomain
from domain.knowledge.enums import DocumentVisibility
from fakes import FakeUnitOfWorkFactory
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher


async def index(factory, text):
    chunks = [RawDocument(page_content=text, metadata={"document_id": 1})] if text else []
    await process_chunks(
        factory,
        1,
        "report.txt",
        chunks,
        DocumentVisibility.INTERNAL_PUBLIC.value,
        owner_id=None,
        group_id=None,
        doc_domain=DocDomain.GENERAL.value,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["new text", ""])
async def test_replacement_enqueues_old_ids_and_preserves_other_documents(replacement):
    factory = FakeUnitOfWorkFactory()
    await index(factory, "old text")
    await factory._uow.chunks.bulk_insert(
        2,
        "other.txt",
        DocumentVisibility.INTERNAL_PRIVATE.value,
        [ChunkInsert(content="other text")],
        owner_id=9,
    )
    await index(factory, replacement)
    entries = list(factory._uow.vector_outbox._entries.values())
    deletes = [e for e in entries if e.operation == OutboxOperation.DELETE_CHUNKS]
    assert [e.payload["chunk_ids"] for e in deletes] == [[1]]
    assert [c["content"] for c in factory._uow.chunks._chunks] == (
        ["other text", "new text"] if replacement else ["other text"]
    )


@pytest.mark.asyncio
async def test_failed_outbox_enqueue_rolls_back_replacement():
    factory = FakeUnitOfWorkFactory()
    await index(factory, "old text")
    with patch.object(factory._uow.vector_outbox, "enqueue", AsyncMock(side_effect=RuntimeError("offline"))):
        with pytest.raises(RuntimeError, match="offline"):
            await index(factory, "new text")
    assert [c["content"] for c in factory._uow.chunks._chunks] == ["old text"]
    assert len(factory._uow.vector_outbox._entries) == 1


@pytest.mark.asyncio
async def test_delayed_old_upsert_cannot_restore_replaced_points():
    factory = FakeUnitOfWorkFactory()
    factory._uow.documents.get_by_id = AsyncMock(return_value=SimpleNamespace(id=1, filename="report.txt"))
    await index(factory, "old text")
    await index(factory, "new text")
    points = {}

    async def upload(chunks, **kwargs):
        points.update({c.metadata["chunk_id"]: c.content for c in chunks})

    async def delete(ids):
        for chunk_id in ids:
            points.pop(chunk_id, None)

    vector = SimpleNamespace(
        ensure_collection=AsyncMock(),
        upload_documents=upload,
        delete_by_ids=delete,
        update_filename_by_document_id=AsyncMock(),
    )
    dispatcher = OutboxDispatcher(factory, vector)
    entries = list(factory._uow.vector_outbox._entries.values())
    for entry in reversed(entries):
        await dispatcher._apply_one(entry)
    assert points == {2: "new text"}
    assert all(e.status == OutboxStatus.DONE for e in entries)


@pytest.mark.asyncio
async def test_replacement_during_upload_removes_inflight_old_points():
    factory = FakeUnitOfWorkFactory()
    factory._uow.documents.get_by_id = AsyncMock(return_value=SimpleNamespace(id=1, filename="report.txt"))
    await index(factory, "old text")
    old_entry = next(iter(factory._uow.vector_outbox._entries.values()))
    points = {}

    async def upload(chunks, **kwargs):
        await index(factory, "new text")
        points.update({c.metadata["chunk_id"]: c.content for c in chunks})

    async def delete(ids):
        for chunk_id in ids:
            points.pop(chunk_id, None)

    vector = SimpleNamespace(
        ensure_collection=AsyncMock(),
        upload_documents=upload,
        delete_by_ids=delete,
        update_filename_by_document_id=AsyncMock(),
    )
    await OutboxDispatcher(factory, vector)._apply_one(old_entry)
    assert points == {}
    assert old_entry.status == OutboxStatus.DONE


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_fails", [False, True])
async def test_single_file_ingestion_commits_outbox_before_recording_success(sync_fails):
    events = []

    async def sync(*args, **kwargs):
        events.append("sync")
        if sync_fails:
            raise ConnectionError("sync unavailable")

    async def record(*args, **kwargs):
        events.append("registry")

    chunk = RawDocument(page_content="new text", metadata={})
    info = SimpleNamespace(
        filename="report.txt",
        key="report.txt",
        extension=".txt",
        size_bytes=8,
        last_modified=datetime(2026, 1, 1),
    )

    workflow = SingleFileIngestionWorkflow(
        SimpleNamespace(get_file_info=AsyncMock(return_value=info), supported_extensions=(".txt",)),
        SimpleNamespace(
            load_file=AsyncMock(return_value=[chunk]),
            classify_text_domain=lambda text, domain: domain,
            index_docs=lambda docs, **kwargs: docs,
            s3_source_key=lambda info: "s3://bucket/report.txt",
        ),
        SimpleNamespace(
            list_all=AsyncMock(return_value={}),
            is_indexed=AsyncMock(return_value=False),
            upsert=record,
        ),
        SimpleNamespace(sync_documents_to_db=sync),
    )
    if sync_fails:
        with pytest.raises(ConnectionError, match="sync unavailable"):
            await workflow.run("report.txt", scope=IngestionScope())
    else:
        await workflow.run("report.txt", scope=IngestionScope())
    assert events == (["sync"] if sync_fails else ["sync", "registry"])


@pytest.mark.asyncio
async def test_failed_inflight_cleanup_is_retried_even_after_document_deletion():
    factory = FakeUnitOfWorkFactory()
    factory._uow.documents.get_by_id = AsyncMock(return_value=SimpleNamespace(id=1, filename="report.txt"))
    await index(factory, "old text")
    entry = next(iter(factory._uow.vector_outbox._entries.values()))
    points = {}
    failures = 1

    async def upload(chunks, **kwargs):
        factory._uow.chunks._chunks.clear()
        factory._uow.documents.get_by_id.return_value = None
        points.update({c.metadata["chunk_id"]: c.content for c in chunks})

    async def delete(ids):
        nonlocal failures
        if failures:
            failures -= 1
            raise ConnectionError("cleanup unavailable")
        for chunk_id in ids:
            points.pop(chunk_id, None)

    vector = SimpleNamespace(
        ensure_collection=AsyncMock(),
        upload_documents=upload,
        delete_by_ids=delete,
        update_filename_by_document_id=AsyncMock(),
    )
    dispatcher = OutboxDispatcher(factory, vector)
    await dispatcher._apply_one(entry)
    assert entry.status == OutboxStatus.FAILED
    assert points == {1: "old text"}
    await dispatcher._apply_one(entry)
    assert points == {}
    assert entry.status == OutboxStatus.DONE
