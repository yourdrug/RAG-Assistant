"""A prefix reset preserves source identity, assignments and unrelated content."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.ingestion import (
    BatchIngestionWorkflow,
    DocumentSyncService,
    IngestionRegistry,
    IngestionService,
    S3DocumentLoader,
    S3IngestionTargets,
    S3UploadService,
    SingleFileIngestionWorkflow,
)
from application.ports.file_storage import FileItem
from domain.entities.chunk import Chunk
from domain.entities.document import Document
from domain.entities.raw_document import RawDocument
from domain.entities.vector_outbox_entry import OutboxOperation, OutboxStatus, VectorOutboxEntry
from domain.value_objects.source_type import SourceType
from domain.value_objects.visibility import DocumentVisibility
from fakes import FakeUnitOfWorkFactory  # type: ignore[import-not-found]
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher


@pytest.fixture
def ingestion(tmp_path: Path) -> SimpleNamespace:
    factory = FakeUnitOfWorkFactory()
    files = {"docs/sub/a.txt": "sub text", "docs/other/a.txt": "other text"}

    def info(key: str) -> FileItem:
        return FileItem(
            key=key,
            filename=key.rsplit("/", 1)[-1],
            extension=".txt",
            size_bytes=len(files[key]),
            last_modified="2026-01-01",
        )

    async def listing(prefix: str) -> list[FileItem]:
        return [info(key) for key in files if key.startswith(prefix)]

    async def download(key: str) -> Path:
        path = tmp_path / "download.txt"
        path.write_text(files[key])
        return path

    storage = SimpleNamespace(
        supported_extensions=(".txt",),
        list_files=listing,
        get_file_info=AsyncMock(side_effect=lambda key: info(key) if key in files else None),
        download_to_temp=AsyncMock(side_effect=download),
    )
    parser = SimpleNamespace(
        parse=lambda path, meta: [
            RawDocument(page_content=path.read_text(), metadata={"source": meta.source_key})
        ]
    )
    splitter = SimpleNamespace(split=lambda docs, _context: docs)
    settings = SimpleNamespace(
        s3_bucket="bucket",
        embed_dim=384,
        document_domain_marker_threshold=1,
        tei_embed_url="tei",
        qdrant_url="qdrant",
        collection_name="chunks",
    )
    registry = IngestionRegistry(factory, storage)
    sync = DocumentSyncService(factory)
    loader = S3DocumentLoader(storage, parser, splitter, settings, registry=registry)
    vector = SimpleNamespace(ensure_collection=AsyncMock())
    targets = S3IngestionTargets()
    service = IngestionService(
        BatchIngestionWorkflow(vector, settings, loader, registry, sync=sync),
        SingleFileIngestionWorkflow(storage, loader, registry, sync),
        registry,
        targets,
        S3UploadService(storage, targets),
    )
    return SimpleNamespace(
        service=service,
        factory=factory,
        files=files,
        registry=registry,
        storage=storage,
        loader=loader,
        vector=vector,
        sync=sync,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["batch", "single"])
async def test_duplicate_basenames_keep_separate_content_and_incremental_state(
    ingestion: SimpleNamespace, mode: str
) -> None:
    env = ingestion

    async def ingest() -> None:
        if mode == "batch":
            await env.service.run_full()
        else:
            for key in env.files:
                await env.service.run_single(key)

    await ingest()
    uow = env.factory._uow
    documents = {doc.ingestion_key: doc for doc in uow.documents._documents.values()}
    assert set(documents) == set(env.files)
    assert len({doc.id for doc in documents.values()}) == 2
    registry = await env.registry.list_all()
    assert set(registry) == set(env.files)

    for key, text in env.files.items():
        assert registry[key]["source"] == f"s3://bucket/{key}"
        rows, _ = await uow.chunks.list_for_document(documents[key].id)
        assert [row.content for row in rows] == [text]

    upserts = [
        entry
        for entry in uow.vector_outbox._entries.values()
        if entry.operation == OutboxOperation.UPSERT_CHUNKS
    ]
    assert {
        (point["metadata"]["source"], point["metadata"]["document_id"])
        for entry in upserts
        for point in entry.payload["points"]
    } == {(f"s3://bucket/{key}", doc.id) for key, doc in documents.items()}

    env.storage.download_to_temp.reset_mock()
    await ingest()
    env.storage.download_to_temp.assert_not_awaited()
    assert await env.registry.list_all() == registry

    env.files["docs/sub/a.txt"] = "updated sub text"
    await ingest()
    env.storage.download_to_temp.assert_awaited_once_with("docs/sub/a.txt")
    assert len(uow.documents._documents) == 2
    assert (await env.registry.list_all())["docs/other/a.txt"] == registry["docs/other/a.txt"]

    for key, text in env.files.items():
        rows, _ = await uow.chunks.list_for_document(documents[key].id)
        assert [row.content for row in rows] == [text]


@pytest.mark.asyncio
async def test_partial_reset_preserves_assignments_other_prefix_and_uploads(
    ingestion: SimpleNamespace,
) -> None:
    env = ingestion
    await env.service.run_full()
    uow = env.factory._uow
    docs = {doc.ingestion_key: doc for doc in uow.documents._documents.values()}
    sub = docs["docs/sub/a.txt"]
    other = docs["docs/other/a.txt"]
    await uow.documents.set_acl(sub.id, DocumentVisibility.CLIENT_PRIVATE, 42, None)
    upload = await uow.documents.save(Document(filename="a.txt", source_path="uploads/a.txt"))
    manual = await uow.documents.save(Document(filename="manual", source_type=SourceType.MANUAL.value))
    other_chunks = [row.copy() for row in uow.chunks._chunks if row["document_id"] == other.id]
    other_registry = (await env.registry.list_all())["docs/other/a.txt"]
    env.files["docs/sub/a.txt"] = "changed sub text"
    await env.service.run_full("docs/sub", reset=True)
    assert set(uow.documents._documents) == {sub.id, other.id, upload.id, manual.id}
    assert sub.owner_id == 42
    assert sub.visibility == DocumentVisibility.CLIENT_PRIVATE
    assert [row for row in uow.chunks._chunks if row["document_id"] == other.id] == other_chunks
    assert (await env.registry.list_all())["docs/other/a.txt"] == other_registry
    sub_chunks = [row for row in uow.chunks._chunks if row["document_id"] == sub.id]
    assert [(row["content"], row["owner_id"], row["visibility"]) for row in sub_chunks] == [
        ("changed sub text", 42, DocumentVisibility.CLIENT_PRIVATE.value)
    ]
    entries = list(uow.vector_outbox._entries.values())
    last = [entry for entry in entries if entry.operation == OutboxOperation.UPSERT_CHUNKS][-1]
    assert last.payload["points"][0]["metadata"]["owner_id"] == 42
    assert last.payload["points"][0]["metadata"]["document_id"] == sub.id
    assert all(entry.operation != OutboxOperation.DELETE_BY_DOCUMENT for entry in entries)
    before = len(entries)
    await env.service.run_full()
    assert len(uow.vector_outbox._entries) == before
    assert all(call.kwargs["reset"] is False for call in env.vector.ensure_collection.await_args_list)


@pytest.mark.asyncio
async def test_one_source_updates_multiple_assignments_without_request_acl(
    ingestion: SimpleNamespace,
) -> None:
    env = ingestion
    await env.service.run_full()
    repo = env.factory._uow.documents
    sub = next(doc for doc in repo._documents.values() if doc.ingestion_key == "docs/sub/a.txt")
    await repo.set_acl(sub.id, DocumentVisibility.INTERNAL_GROUP, None, 7)
    private = await repo.save(
        Document(
            filename="a.txt",
            ingestion_key=sub.ingestion_key,
            visibility=DocumentVisibility.CLIENT_PRIVATE,
            owner_id=42,
        )
    )
    await env.service.run_full("docs/sub/", reset=True, visibility=DocumentVisibility.CLIENT_PRIVATE)
    rows = env.factory._uow.chunks._chunks
    assert [(row["owner_id"], row["group_id"]) for row in rows if row["document_id"] == sub.id] == [(None, 7)]
    assert [(row["owner_id"], row["group_id"]) for row in rows if row["document_id"] == private.id] == [
        (42, None)
    ]


@pytest.mark.asyncio
async def test_empty_reset_removes_missing_sources_and_keeps_literal_prefix_boundaries(
    ingestion: SimpleNamespace,
) -> None:
    env = ingestion
    env.files.update(
        {"docs/submarine/a.txt": "boundary", "docs/Sub/a.txt": "case", "docs/s_b/a.txt": "underscore"}
    )
    await env.service.run_full()
    env.files.pop("docs/sub/a.txt")
    await env.service.run_full("docs/sub/", reset=True)
    assert set(await env.registry.list_all()) == set(env.files)
    assert {doc.ingestion_key for doc in env.factory._uow.documents._documents.values()} == set(env.files)
    deletes = [
        entry
        for entry in env.factory._uow.vector_outbox._entries.values()
        if entry.operation == OutboxOperation.DELETE_BY_DOCUMENT
    ]
    assert len(deletes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["parse", "download", "split", "versioning"])
async def test_preparation_failure_does_not_change_documents_or_chunks(
    ingestion: SimpleNamespace, failure: str
) -> None:
    env = ingestion
    await env.service.run_full()
    uow = env.factory._uow
    before_docs = dict(uow.documents._documents)
    before_chunks = [row.copy() for row in uow.chunks._chunks]
    before_outbox = dict(uow.vector_outbox._entries)

    if failure == "parse":
        env.loader._parser.parse = MagicMock(side_effect=ValueError("parse failed"))
    elif failure == "download":
        env.storage.download_to_temp = AsyncMock(side_effect=OSError("download failed"))
    elif failure == "split":
        env.loader._splitter.split = MagicMock(return_value=[])
    else:
        env.sync.prepare_versioning = AsyncMock(side_effect=ValueError("versioning failed"))

    with pytest.raises((ValueError, OSError)):
        await env.service.run_full("docs/", reset=True)
    assert uow.documents._documents == before_docs
    assert uow.chunks._chunks == before_chunks
    assert uow.vector_outbox._entries == before_outbox


@pytest.mark.asyncio
async def test_force_and_cache_use_full_key_for_duplicate_basenames(ingestion: SimpleNamespace) -> None:
    env = ingestion
    await env.service.run_full()
    before = len(env.factory._uow.vector_outbox._entries)
    await env.service.run_single("docs/sub/a.txt")
    assert len(env.factory._uow.vector_outbox._entries) == before
    await env.service.run_single("docs/sub/a.txt", force=True)
    assert len(env.factory._uow.vector_outbox._entries) > before
    assert set(await env.registry.list_all()) == set(env.files)
    result = await env.service.get_registry()
    assert [item.filename for item in result.files] == ["a.txt", "a.txt"]
    assert {item.ingestion_key for item in result.files} == set(env.files)


@pytest.mark.asyncio
async def test_failed_replacement_clears_cache_and_retry_finishes_all_targets(
    ingestion: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ingestion
    await env.service.run_full()
    uow = env.factory._uow
    before = [row.copy() for row in uow.chunks._chunks]
    enqueue = uow.vector_outbox.enqueue

    async def fail(entry: VectorOutboxEntry) -> VectorOutboxEntry:
        if entry.operation == OutboxOperation.UPSERT_CHUNKS:
            message = "outbox offline"
            raise RuntimeError(message)

        await enqueue(entry)
        return entry

    with monkeypatch.context() as patch:
        patch.setattr(uow.vector_outbox, "enqueue", fail)
        with pytest.raises(RuntimeError, match="outbox offline"):
            await env.service.run_full("docs/sub/", reset=True)
    assert uow.chunks._chunks == before
    assert "docs/sub/a.txt" not in await env.registry.list_all()
    assert "docs/other/a.txt" in await env.registry.list_all()
    await env.service.run_full("docs/sub/", reset=True)
    assert set(await env.registry.list_all()) == set(env.files)


@pytest.mark.asyncio
async def test_delayed_legacy_source_patch_cannot_move_points_between_assignments(
    ingestion: SimpleNamespace,
) -> None:
    env = ingestion
    await env.service.run_full()
    vector = SimpleNamespace(set_document_id_by_source=AsyncMock())
    dispatcher = OutboxDispatcher(env.factory, vector)
    entry = VectorOutboxEntry(
        operation=OutboxOperation.SET_DOCUMENT_ID,
        aggregate_id=1,
        payload={"source": "s3://bucket/docs/sub/a.txt", "document_id": 1},
    )
    await env.factory._uow.vector_outbox.enqueue(entry)
    await dispatcher._apply_one(entry)
    await dispatcher._apply_one(entry)
    assert entry.status == OutboxStatus.DONE
    vector.set_document_id_by_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_outbox_projection_keeps_other_prefix_and_ignores_delayed_replaced_chunks(
    ingestion: SimpleNamespace,
) -> None:
    env = ingestion
    points: dict[int, dict[str, object]] = {}

    async def upload(chunks: list[Chunk], **_kwargs: object) -> None:
        for chunk in chunks:
            points[chunk.metadata["chunk_id"]] = dict(chunk.metadata)

    async def delete(ids: list[int]) -> None:
        for chunk_id in ids:
            points.pop(chunk_id, None)

    vector = SimpleNamespace(
        ensure_collection=AsyncMock(),
        upload_documents=upload,
        delete_by_ids=delete,
        update_filename_by_document_id=AsyncMock(),
    )
    dispatcher = OutboxDispatcher(env.factory, vector)
    await env.service.run_full()
    uow = env.factory._uow
    original_entries = list(uow.vector_outbox._entries.values())

    for entry in original_entries:
        await dispatcher._apply_one(entry)

    other_id = next(
        doc.id for doc in uow.documents._documents.values() if doc.ingestion_key == "docs/other/a.txt"
    )
    other_points = {key: value.copy() for key, value in points.items() if value["document_id"] == other_id}
    env.files["docs/sub/a.txt"] = "replacement"
    await env.service.run_full("docs/sub/", reset=True)

    # Deliver in reverse order, including retries of already applied old jobs.
    for entry in reversed(list(uow.vector_outbox._entries.values())):
        await dispatcher._apply_one(entry)

    assert {key: value for key, value in points.items() if value["document_id"] == other_id} == other_points
    assert set(points) == {row["id"] for row in uow.chunks._chunks}


@pytest.mark.asyncio
async def test_ingestion_source_path_can_be_read_by_storage(ingestion: SimpleNamespace) -> None:
    await ingestion.service.run_full()

    for document in ingestion.factory._uow.documents._documents.values():
        path = await ingestion.storage.download_to_temp(document.source_path)
        assert path.read_text() == ingestion.files[document.ingestion_key]
        path.unlink()


def configure_upload_race(
    env: SimpleNamespace,
) -> tuple[dict[str, bytes], asyncio.Event, asyncio.Event, asyncio.Event]:
    objects = {"docs/a.txt": b"original"}
    first_written, release, second_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def upload(key: str, data: bytes) -> None:
        objects[key] = data

        if data == b"first":
            first_written.set()
            await release.wait()
            msg = "response lost"
            raise OSError(msg)

    async def copy(old: str, new: str) -> None:
        objects[new] = objects[old]

    async def delete(key: str) -> None:
        objects.pop(key, None)

    env.storage.upload_file = upload
    env.storage.copy_file = copy
    env.storage.delete_file = delete
    env.storage.get_file_info = AsyncMock(side_effect=lambda key: object() if key in objects else None)

    return objects, first_written, release, second_entered


@pytest.mark.asyncio
async def test_upload_rollback_cannot_overwrite_another_request(ingestion: SimpleNamespace) -> None:
    env = ingestion
    lock = asyncio.Lock()
    env.registry.ingestion_lock = lambda: lock
    objects, first_written, release, second_entered = configure_upload_race(env)

    # Separate facade instances must share the registry's cross-process lock.
    second_service = IngestionService(
        MagicMock(),
        MagicMock(),
        env.registry,
        S3IngestionTargets(),
        S3UploadService(env.storage, S3IngestionTargets()),
    )

    async def upload_second_service() -> list[str]:
        second_entered.set()
        return await second_service.upload_files([SimpleNamespace(filename="a.txt", data=b"second")])

    first = asyncio.create_task(env.service.upload_files([SimpleNamespace(filename="a.txt", data=b"first")]))
    second = None
    try:
        await asyncio.wait_for(first_written.wait(), 5)
        second = asyncio.create_task(upload_second_service())
        await asyncio.wait_for(second_entered.wait(), 5)
    finally:
        release.set()
        results = await asyncio.wait_for(
            asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True), 5
        )

    assert isinstance(results[0], OSError)
    assert results[1] == ["docs/a.txt"]
    assert objects == {"docs/a.txt": b"second"}
