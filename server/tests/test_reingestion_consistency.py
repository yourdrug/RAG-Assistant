"""Replacement must converge SQL, vector outbox and the sparse corpus."""

from types import SimpleNamespace
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from application.services.document_pipeline import process_chunks
from application.services.single_file_ingestion import SingleFileIngestionWorkflow
from application.services.ingestion_scope import IngestionScope
from domain.entities.raw_document import RawDocument
from domain.entities.vector_outbox_entry import OutboxOperation, OutboxStatus
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.visibility import DocumentVisibility
from fakes import FakeUnitOfWorkFactory
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.sparse_index_admin import S3SparseIndexAdmin
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher


async def index(factory, text):
    chunks = [RawDocument(page_content=text, metadata={"document_id": 1})] if text else []
    await process_chunks(
        factory,
        1,
        "report.txt",
        chunks,
        DocumentVisibility.INTERNAL_PUBLIC.value,
        None,
        None,
        DocDomain.GENERAL.value,
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
        ["other text"],
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
    factory._uow.documents.get_by_id = AsyncMock(return_value=SimpleNamespace(id=1))
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
    )
    dispatcher = OutboxDispatcher(factory, vector)
    entries = list(factory._uow.vector_outbox._entries.values())
    for entry in reversed(entries):
        await dispatcher._apply_one(entry)
    assert points == {2: "new text"}
    assert all(e.status == OutboxStatus.DONE for e in entries)


@pytest.mark.asyncio
async def test_sparse_refresh_uses_current_sql_corpus_with_acl():
    factory = FakeUnitOfWorkFactory()
    await index(factory, "old text")
    await factory._uow.chunks.bulk_insert(
        2,
        "other.txt",
        DocumentVisibility.INTERNAL_PRIVATE.value,
        ["other text"],
        owner_id=9,
    )
    await index(factory, "new text")
    saved = []

    async def save(index, storage):
        saved.append(BM25Index.from_dict(index.to_dict()))

    with (
        patch("infrastructure.bm25.sparse_index_admin.save_bm25_index_to_s3", side_effect=save),
        patch(
            "infrastructure.bm25.sparse_index_admin.publish_bm25_invalidation",
            new=AsyncMock(),
        ),
    ):
        admin = S3SparseIndexAdmin(object(), factory)
        await admin.extend([])
        await admin.extend([])
    assert [idx.texts for idx in saved] == [["other text", "new text"]] * 2
    assert saved[-1].doc_visibility == [
        DocumentVisibility.INTERNAL_PRIVATE.value,
        DocumentVisibility.INTERNAL_PUBLIC.value,
    ]
    assert saved[-1].doc_owner_id == [9, None]


@pytest.mark.asyncio
async def test_replacement_during_upload_removes_inflight_old_points():
    factory = FakeUnitOfWorkFactory()
    factory._uow.documents.get_by_id = AsyncMock(return_value=SimpleNamespace(id=1))
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
    )
    await OutboxDispatcher(factory, vector)._apply_one(old_entry)
    assert points == {}
    assert old_entry.status == OutboxStatus.DONE


@pytest.mark.asyncio
@pytest.mark.parametrize("sparse_fails", [False, True])
async def test_single_file_ingestion_refreshes_sparse_before_recording_success(sparse_fails):
    events = []

    async def sync(*args, **kwargs):
        events.append("sync")

    async def refresh(*args, **kwargs):
        events.append("sparse")
        if sparse_fails:
            raise ConnectionError("sparse unavailable")

    async def record(*args, **kwargs):
        events.append("registry")

    chunk = RawDocument(page_content="new text", metadata={})
    info = SimpleNamespace(
        filename="report.txt",
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
        sparse_index=SimpleNamespace(update=refresh),
    )
    if sparse_fails:
        with pytest.raises(ConnectionError, match="sparse unavailable"):
            await workflow.run("report.txt", scope=IngestionScope())
    else:
        await workflow.run("report.txt", scope=IngestionScope())
    assert events == (["sync", "sparse"] if sparse_fails else ["sync", "sparse", "registry"])


@pytest.mark.asyncio
async def test_failed_inflight_cleanup_is_retried_even_after_document_deletion():
    factory = FakeUnitOfWorkFactory()
    factory._uow.documents.get_by_id = AsyncMock(return_value=SimpleNamespace(id=1))
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
    )
    dispatcher = OutboxDispatcher(factory, vector)
    await dispatcher._apply_one(entry)
    assert entry.status == OutboxStatus.FAILED
    assert points == {1: "old text"}
    await dispatcher._apply_one(entry)
    assert points == {}
    assert entry.status == OutboxStatus.DONE
