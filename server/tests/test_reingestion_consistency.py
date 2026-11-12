"""Replacement must converge SQL, vector outbox and the sparse corpus."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.indexing import process_chunks
from application.ingestion import IngestionScope, SingleFileIngestionWorkflow
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.enums import DocDomain, DocumentVisibility
from domain.knowledge.index_sync import OutboxOperation, OutboxStatus, VectorOutboxEntry
from domain.knowledge.repositories.chunk import ChunkInsertOptions
from domain.shared.hashing import content_hash
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from domain.knowledge.chunk import Chunk


async def index(factory: FakeUnitOfWorkFactory, text: str) -> None:
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
@pytest.mark.parametrize("cancel_upload", [False, True])
async def test_document_delete_waits_for_inflight_upload_before_completing(cancel_upload: bool) -> None:  # noqa: PLR0915 - Complete worker cancellation/deletion race.
    factory = FakeUnitOfWorkFactory()
    await index(factory, "old text")
    upload_started, release_upload, deletion_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    points = {}

    async def upload(chunks: list[Chunk], **kwargs: Any) -> None:
        upload_started.set()
        await release_upload.wait()
        points.update({chunk.metadata["chunk_id"]: chunk.content for chunk in chunks})

    async def delete_ids(ids: list[int]) -> None:
        for chunk_id in ids:
            points.pop(chunk_id, None)

    async def delete_document(document_id: int) -> None:
        points.clear()

    vector = SimpleNamespace(
        ensure_collection=AsyncMock(),
        upload_documents=upload,
        delete_by_ids=delete_ids,
        delete_by_document_id=delete_document,
        update_filename_by_document_id=AsyncMock(),
    )
    # Independent workers share the per-document lock, as separate processes do in PostgreSQL.
    deletion = None
    uploading = asyncio.create_task(
        OutboxDispatcher(factory, vector)._apply_one(
            next(iter(factory._uow.fake_vector_outbox._entries.values()))
        )
    )
    try:
        await asyncio.wait_for(upload_started.wait(), 5)

        if cancel_upload:
            uploading.cancel()
            await asyncio.sleep(0)

        entry = await factory._uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.DELETE_BY_DOCUMENT,
                aggregate_id=1,
                payload={"document_id": 1},
            )
        )

        async def dispatch_delete() -> None:
            deletion_started.set()
            await OutboxDispatcher(factory, vector)._apply_one(entry)

        deletion = asyncio.create_task(dispatch_delete())
        await asyncio.wait_for(deletion_started.wait(), 5)
        assert entry.status != OutboxStatus.DONE
    finally:
        factory._uow.fake_chunks._chunks.clear()
        release_upload.set()
        await asyncio.wait_for(
            asyncio.gather(uploading, *([deletion] if deletion is not None else []), return_exceptions=True),
            5,
        )

    assert (entry.status, points) == (OutboxStatus.DONE, {})
    assert uploading.cancelled() is cancel_upload


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["new text", ""])
async def test_replacement_enqueues_old_ids_and_preserves_other_documents(replacement: str) -> None:
    factory = FakeUnitOfWorkFactory()
    await index(factory, "old text")
    await factory._uow.chunks.bulk_insert(
        2,
        "other.txt",
        DocumentVisibility.INTERNAL_PRIVATE.value,
        [ChunkInsert(content="other text")],
        options=ChunkInsertOptions(owner_id=9),
    )
    await index(factory, replacement)
    entries = list(factory._uow.fake_vector_outbox._entries.values())
    deletes = [e for e in entries if e.operation == OutboxOperation.DELETE_CHUNKS]
    assert [e.payload["chunk_ids"] for e in deletes] == [[1]]
    assert [c["content"] for c in factory._uow.fake_chunks._chunks] == (
        ["other text", "new text"] if replacement else ["other text"]
    )


@pytest.mark.asyncio
async def test_failed_outbox_enqueue_rolls_back_replacement() -> None:
    factory = FakeUnitOfWorkFactory()
    await index(factory, "old text")
    with (
        patch.object(factory._uow.vector_outbox, "enqueue", AsyncMock(side_effect=RuntimeError("offline"))),
        pytest.raises(RuntimeError, match="offline"),
    ):
        await index(factory, "new text")
    assert [c["content"] for c in factory._uow.fake_chunks._chunks] == ["old text"]
    assert len(factory._uow.fake_vector_outbox._entries) == 1


@pytest.mark.asyncio
async def test_delayed_old_upsert_cannot_restore_replaced_points(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = FakeUnitOfWorkFactory()
    monkeypatch.setattr(
        factory._uow.documents,
        "get_by_id",
        AsyncMock(return_value=SimpleNamespace(id=1, filename="report.txt")),
    )
    await index(factory, "old text")
    await index(factory, "new text")
    points = {}

    async def upload(chunks: list[Chunk], **kwargs: Any) -> None:
        points.update({c.metadata["chunk_id"]: c.content for c in chunks})

    async def delete(ids: list[int]) -> None:
        for chunk_id in ids:
            points.pop(chunk_id, None)

    vector = SimpleNamespace(
        ensure_collection=AsyncMock(),
        upload_documents=upload,
        delete_by_ids=delete,
        update_filename_by_document_id=AsyncMock(),
    )
    dispatcher = OutboxDispatcher(factory, vector)
    entries = list(factory._uow.fake_vector_outbox._entries.values())

    for entry in reversed(entries):
        await dispatcher._apply_one(entry)

    assert points == {2: "new text"}
    assert all(e.status == OutboxStatus.DONE for e in entries)


@pytest.mark.asyncio
async def test_replacement_during_upload_removes_inflight_old_points(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = FakeUnitOfWorkFactory()
    monkeypatch.setattr(
        factory._uow.documents,
        "get_by_id",
        AsyncMock(return_value=SimpleNamespace(id=1, filename="report.txt")),
    )
    await index(factory, "old text")
    old_entry = next(iter(factory._uow.fake_vector_outbox._entries.values()))
    points = {}

    async def upload(chunks: list[Chunk], **kwargs: Any) -> None:
        await index(factory, "new text")
        points.update({c.metadata["chunk_id"]: c.content for c in chunks})

    async def delete(ids: list[int]) -> None:
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
    assert OutboxStatus(old_entry.status) == OutboxStatus.DONE


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_fails", [False, True])
async def test_single_file_ingestion_commits_outbox_before_recording_success(sync_fails: bool) -> None:
    events = []

    async def sync(*args: object, **kwargs: Any) -> None:
        events.append("sync")

        if sync_fails:
            message = "sync unavailable"
            raise ConnectionError(message)

    async def record(*args: object, **kwargs: Any) -> None:
        events.append("registry")

    chunk = RawDocument(page_content="new text", metadata={})
    info = SimpleNamespace(
        filename="report.txt",
        key="report.txt",
        extension=".txt",
        size_bytes=8,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
    )

    workflow = SingleFileIngestionWorkflow(
        SimpleNamespace(get_file_info=AsyncMock(return_value=info), supported_extensions=(".txt",)),
        MagicMock(
            load_file=AsyncMock(return_value=[chunk]),
            classify_text_domain=lambda _text, domain: domain,
            index_docs=lambda docs, **_kwargs: docs,
            s3_source_key=lambda _info: "s3://bucket/report.txt",
        ),
        MagicMock(
            list_all=AsyncMock(return_value={}),
            is_indexed=AsyncMock(return_value=False),
            upsert=record,
        ),
        MagicMock(sync_documents_to_db=sync),
    )

    if sync_fails:
        with pytest.raises(ConnectionError, match="sync unavailable"):
            await workflow.run("report.txt", scope=IngestionScope())
    else:
        await workflow.run("report.txt", scope=IngestionScope())

    assert events == (["sync"] if sync_fails else ["sync", "registry"])


@pytest.mark.asyncio
async def test_failed_inflight_cleanup_is_retried_even_after_document_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeUnitOfWorkFactory()
    monkeypatch.setattr(
        factory._uow.documents,
        "get_by_id",
        AsyncMock(return_value=SimpleNamespace(id=1, filename="report.txt")),
    )
    await index(factory, "old text")
    entry = next(iter(factory._uow.fake_vector_outbox._entries.values()))
    points = {}
    failures = 1

    async def upload(chunks: list[Chunk], **kwargs: Any) -> None:
        factory._uow.fake_chunks._chunks.clear()
        cast("AsyncMock", factory._uow.documents.get_by_id).return_value = None
        points.update({c.metadata["chunk_id"]: c.content for c in chunks})

    async def delete(ids: list[int]) -> None:
        nonlocal failures

        if failures:
            failures -= 1
            message = "cleanup unavailable"
            raise ConnectionError(message)

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
    assert OutboxStatus(entry.status) == OutboxStatus.DONE


@pytest.mark.asyncio
async def test_delayed_upsert_uses_current_content_of_same_chunk() -> None:
    factory = FakeUnitOfWorkFactory()
    await index(factory, "old text")
    entry = next(iter(factory._uow.fake_vector_outbox._entries.values()))
    factory._uow.fake_chunks._chunks[0].update(content="new text", content_hash=content_hash("new text"))
    points = {}

    async def upload(chunks: list[Chunk], **kwargs: Any) -> None:
        points.update({c.metadata["chunk_id"]: (c.content, c.metadata.get("content_hash")) for c in chunks})

    vector = SimpleNamespace(
        ensure_collection=AsyncMock(),
        upload_documents=upload,
        delete_by_ids=AsyncMock(),
        update_filename_by_document_id=AsyncMock(),
    )
    await OutboxDispatcher(factory, vector)._apply_one(entry)
    assert points == {1: ("new text", content_hash("new text"))}
    assert OutboxStatus(entry.status) == OutboxStatus.DONE


@pytest.mark.asyncio
async def test_overlapping_upserts_retry_stale_write_with_current_content() -> None:
    factory = FakeUnitOfWorkFactory()
    await index(factory, "old text")
    old_entry = next(iter(factory._uow.fake_vector_outbox._entries.values()))
    old_started = asyncio.Event()
    new_uploaded = asyncio.Event()
    points = {}

    async def upload(chunks: list[Chunk], **kwargs: Any) -> None:
        content = chunks[0].content

        if content == "old text" and not old_started.is_set():
            old_started.set()
            await new_uploaded.wait()

        points.update({c.metadata["chunk_id"]: c.content for c in chunks})

    vector = SimpleNamespace(
        ensure_collection=AsyncMock(),
        upload_documents=upload,
        delete_by_ids=AsyncMock(),
        update_filename_by_document_id=AsyncMock(),
    )
    first, second = OutboxDispatcher(factory, vector), OutboxDispatcher(factory, vector)
    async with asyncio.timeout(2), asyncio.TaskGroup() as group:
        first_task = group.create_task(first._apply_one(old_entry))
        await old_started.wait()
        factory._uow.fake_chunks._chunks[0].update(content="new text", content_hash=content_hash("new text"))
        second_task = group.create_task(second.apply_upsert(old_entry.payload, document_id=1))
        await asyncio.sleep(0)
        assert not second_task.done()
        new_uploaded.set()

    assert first_task.done()
    assert old_entry.status == OutboxStatus.FAILED
    await first._apply_one(old_entry)
    assert points == {1: "new text"}
    assert OutboxStatus(old_entry.status) == OutboxStatus.DONE
