"""RAG-002: complete object keys identify ingestion sources independently."""

import asyncio
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from application.services.ingest_service import IngestAppService
from application.services.ingestion_source import object_key_from_source
from application.services.ingestion_sync import DocumentSyncService
from domain.entities.document import Document
from domain.entities.vector_outbox_entry import OutboxOperation
from domain.exceptions import ValidationError
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.visibility import DocumentVisibility
from presentation.cli.commands import ingest as ingest_cli
from ingestion_helpers import (
    FIRST_KEY,
    SECOND_KEY,
    FIRST_SOURCE,
    SECOND_SOURCE,
    FILENAME,
    ingestion_components,
    source_chunks,
    sync_entries,
)


@pytest.mark.parametrize("source", ["manual.md", "s3:///manual.md", "s3://bucket/", "s3://bucket"])
def test_source_identity_rejects_missing_bucket_or_key(source):
    with pytest.raises(ValidationError):
        object_key_from_source(source)


def test_source_identity_preserves_percent_characters_and_long_key():
    key = "docs/" + "section/" * 100 + "manual%2F #?.md"
    assert object_key_from_source(f"s3://bucket/{key}") == key


def test_cli_list_displays_full_keys_with_registry_timestamps(tmp_path, monkeypatch, caplog):
    service, _, _, _ = ingestion_components(tmp_path)
    asyncio.run(service.run_full_ingestion("docs/", domain=DocDomain.GENERAL.value))

    async def run_with_test_infrastructure(operation):
        return await operation(None)

    monkeypatch.setattr(ingest_cli, "_run_with_infrastructure", run_with_test_infrastructure)
    monkeypatch.setattr(ingest_cli, "_create_service", lambda infrastructure: service)
    monkeypatch.setattr(ingest_cli.logger, "disabled", False)
    ingest_cli.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=ingest_cli.logger.name):
            result = CliRunner().invoke(ingest_cli.ingest_app, ["list"])
    finally:
        ingest_cli.logger.removeHandler(caplog.handler)
    assert result.exit_code == 0
    assert FIRST_KEY in caplog.text
    assert SECOND_KEY in caplog.text


def test_cli_force_reindexes_only_requested_source(tmp_path, monkeypatch):
    service, factory, _, _ = ingestion_components(tmp_path)
    asyncio.run(service.run_full_ingestion("docs/", domain=DocDomain.GENERAL.value))
    count = len(factory._uow.vector_outbox._entries)

    async def run_with_test_infrastructure(operation):
        return await operation(None)

    monkeypatch.setattr(ingest_cli, "_run_with_infrastructure", run_with_test_infrastructure)
    monkeypatch.setattr(ingest_cli, "_create_service", lambda infrastructure: service)
    result = CliRunner().invoke(
        ingest_cli.ingest_app, ["file", FIRST_KEY, "--force", "--domain", DocDomain.GENERAL.value]
    )
    assert result.exit_code == 0
    new_upserts = [
        entry
        for entry in list(factory._uow.vector_outbox._entries.values())[count:]
        if entry.operation == OutboxOperation.UPSERT_CHUNKS
    ]
    assert len(new_upserts) == 1
    assert new_upserts[0].payload["points"][0]["metadata"]["source"] == FIRST_SOURCE


def test_collect_entries_keeps_both_same_name_sources(tmp_path):
    service, _, _, _ = ingestion_components(tmp_path)
    chunks = source_chunks()
    entries = service._registry.collect_entries(chunks, chunks, {FIRST_SOURCE: 10, SECOND_SOURCE: 10})

    assert set(entries) == {FIRST_KEY, SECOND_KEY}
    assert entries[FIRST_KEY]["filename"] == FILENAME
    assert entries[SECOND_KEY]["source"] == SECOND_SOURCE
    assert entries[FIRST_KEY]["hash"] == entries[SECOND_KEY]["hash"]


@pytest.mark.asyncio
async def test_single_file_equal_hash_does_not_skip_other_same_name_source(tmp_path):
    service, factory, _, _ = ingestion_components(tmp_path)
    await service.run_single_file(FIRST_KEY, domain=DocDomain.GENERAL.value)
    await service.run_single_file(SECOND_KEY, domain=DocDomain.GENERAL.value)

    assert set(await service.get_registry()) == {FIRST_KEY, SECOND_KEY}
    documents = await factory._uow.documents.list_all()
    assert {doc.filename for doc in documents} == {"manual.md", "manual(1).md"}
    assert {getattr(doc, "ingestion_key", None) for doc in documents} == {FIRST_KEY, SECOND_KEY}
    assert len({doc.id for doc in documents}) == 2
    upserts = [
        entry
        for entry in factory._uow.vector_outbox._entries.values()
        if entry.operation == OutboxOperation.UPSERT_CHUNKS
    ]
    assert {entry.payload["points"][0]["metadata"]["source"] for entry in upserts} == {
        FIRST_SOURCE,
        SECOND_SOURCE,
    }
    assert len({entry.aggregate_id for entry in upserts}) == 2


@pytest.mark.asyncio
async def test_batch_same_name_sources_both_reach_sql_and_sparse(tmp_path):
    service, factory, _, sparse = ingestion_components(tmp_path)
    await service.run_full_ingestion("docs/", domain=DocDomain.GENERAL.value)

    documents = await factory._uow.documents.list_all()
    assert len(documents) == 2
    texts = []
    for doc in documents:
        chunks, _ = await factory._uow.chunks.list_for_document(doc.id)
        texts.extend(chunk.content for chunk in chunks)
    assert sorted(texts) == ["first text", "other text"]
    assert sorted(sparse.texts) == sorted(texts)
    assert set(await service.get_registry()) == {FIRST_KEY, SECOND_KEY}
    count = len(factory._uow.vector_outbox._entries)
    await service.run_full_ingestion("docs/", domain=DocDomain.GENERAL.value)
    assert len(factory._uow.vector_outbox._entries) == count


@pytest.mark.asyncio
async def test_force_reindex_only_invalidates_exact_key(tmp_path):
    service, _, _, _ = ingestion_components(tmp_path)
    await service.run_single_file(FIRST_KEY, domain=DocDomain.GENERAL.value)
    await service.run_single_file(SECOND_KEY, domain=DocDomain.GENERAL.value)
    await service.force_reindex(FIRST_KEY)

    assert set(await service.get_registry()) == {SECOND_KEY}


@pytest.mark.asyncio
async def test_registry_api_keeps_basename_and_distinct_source(tmp_path):
    service, factory, _, _ = ingestion_components(tmp_path)
    await service.run_full_ingestion("docs/", domain=DocDomain.GENERAL.value)
    result = await IngestAppService(factory, service).get_registry()

    assert result.total_files == 2
    assert [item.filename for item in result.files] == [FILENAME, FILENAME]
    assert {item.source for item in result.files} == {FIRST_SOURCE, SECOND_SOURCE}


@pytest.mark.asyncio
async def test_replacement_and_registry_failure_reuse_bound_document(tmp_path, monkeypatch):
    service, factory, storage, _ = ingestion_components(tmp_path)
    await service.run_full_ingestion("docs/", domain=DocDomain.GENERAL.value)
    before = {doc.ingestion_key: doc.id for doc in await factory._uow.documents.list_all()}
    storage.contents[FIRST_KEY] = "changed text"
    storage.revisions[FIRST_KEY] = "new-mtime"
    original_upsert = factory._uow.ingestion_registry.upsert

    async def fail_registry(entry):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(factory._uow.ingestion_registry, "upsert", fail_registry)
    with pytest.raises(RuntimeError, match="registry unavailable"):
        await service.run_single_file(FIRST_KEY, domain=DocDomain.GENERAL.value)
    monkeypatch.setattr(factory._uow.ingestion_registry, "upsert", original_upsert)
    await service.run_single_file(FIRST_KEY, domain=DocDomain.GENERAL.value)

    after = {doc.ingestion_key: doc.id for doc in await factory._uow.documents.list_all()}
    assert after == before
    first_chunks, _ = await factory._uow.chunks.list_for_document(before[FIRST_KEY])
    other_chunks, _ = await factory._uow.chunks.list_for_document(before[SECOND_KEY])
    assert [chunk.content for chunk in first_chunks] == ["changed text"]
    assert [chunk.content for chunk in other_chunks] == ["other text"]


@pytest.mark.asyncio
async def test_sync_does_not_reuse_unrelated_same_name_document(tmp_path):
    _, factory, _, _ = ingestion_components(tmp_path)
    unrelated = await factory._uow.documents.save(
        Document(filename=FILENAME, source_path="uploads/public/1_manual.md")
    )
    await DocumentSyncService(factory).sync_documents_to_db(
        sync_entries((FIRST_KEY,)), {}, source_chunks((FIRST_KEY,))
    )

    documents = await factory._uow.documents.list_all()
    assert len(documents) == 2
    assert getattr(unrelated, "ingestion_key", None) is None
    imported = next(doc for doc in documents if doc.id != unrelated.id)
    assert imported.filename == "manual(1).md"
    assert imported.ingestion_key == FIRST_KEY


@pytest.mark.asyncio
async def test_group_scopes_keep_source_bindings_separate(tmp_path):
    _, factory, _, _ = ingestion_components(tmp_path)
    sync = DocumentSyncService(factory)
    for group_id in (7, 8, 7):
        await sync.sync_documents_to_db(
            sync_entries((FIRST_KEY,)),
            {},
            source_chunks((FIRST_KEY,)),
            visibility=DocumentVisibility.INTERNAL_GROUP,
            group_id=group_id,
        )
    documents = await factory._uow.documents.list_all()
    assert len(documents) == 2
    assert {doc.group_id for doc in documents} == {7, 8}
    assert {getattr(doc, "ingestion_key", None) for doc in documents} == {FIRST_KEY}


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["docs/раздел/manual #?.md", "docs/a//manual.md"])
async def test_registry_preserves_exact_object_key(tmp_path, key):
    service, _, storage, _ = ingestion_components(tmp_path)
    storage.contents = {key: "first text"}
    storage.revisions = {key: "same-mtime"}
    await service.run_full_ingestion("docs/", domain=DocDomain.GENERAL.value)

    entries = await service.get_registry()
    assert list(entries) == [key]
    assert entries[key]["filename"] == Path(key).name
