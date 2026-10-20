"""Behavioral contracts for the reusable ingestion workflows."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest
from langchain.schema import Document

from application.ingestion import BatchIngestionWorkflow
from application.ingestion import IngestionScope
from application.ingestion import SingleFileIngestionWorkflow
from domain.knowledge.enums import DocumentVisibility


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["docs/sub/", "other/", "docs/"])
async def test_empty_reset_prunes_selected_prefix_without_resetting_collection(prefix):
    vector_store = MagicMock(ensure_collection=AsyncMock())
    settings = MagicMock(embed_dim=384)
    loader = MagicMock(load_documents=AsyncMock(return_value=([], 0)))
    registry = MagicMock(list_all=AsyncMock(return_value={}))
    sync = MagicMock(delete_missing_sources=AsyncMock())
    workflow = BatchIngestionWorkflow(vector_store, settings, loader, registry, sync=sync)
    await workflow.run(prefix, reset=True, scope=IngestionScope())
    vector_store.ensure_collection.assert_awaited_once_with(384, reset=False)
    sync.delete_missing_sources.assert_awaited_once_with(prefix, set())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "sync"])
async def test_batch_workflow_syncs_before_recording_registry_entry(failure):
    source = "s3://bucket/docs/report.pdf"
    document = Document(page_content="report text", metadata={"source": source})
    chunk = Document(page_content="report text", metadata={"source": source})
    events: list[str] = []

    vector_store = MagicMock()
    vector_store.ensure_collection = AsyncMock()
    settings = MagicMock(
        embed_dim=384,
        s3_bucket="bucket",
        tei_embed_url="http://tei",
        qdrant_url="http://qdrant",
        collection_name="chunks",
    )
    loader = MagicMock()
    loader.load_documents = AsyncMock(return_value=([document], 0))
    loader.split_docs.return_value = [chunk]
    loader.classify_source_domains.return_value = {source: "general"}
    registry = MagicMock()
    registry.list_all = AsyncMock(return_value={})
    registry.collect_entries = AsyncMock(
        return_value={"report.pdf": {"hash": "hash", "source": source, "chunks": 1, "chars": 11}}
    )

    async def persist_entries(entries):
        events.append("registry")

    registry.persist_entries = AsyncMock(side_effect=persist_entries)
    sync = MagicMock()

    async def sync_documents(*args, **kwargs):
        events.append("sync")
        if failure == "sync":
            raise ConnectionError("sync unavailable")

    sync.sync_documents_to_db = AsyncMock(side_effect=sync_documents)
    workflow = BatchIngestionWorkflow(vector_store, settings, loader, registry, sync=sync)
    scope = IngestionScope(
        visibility=DocumentVisibility.INTERNAL_GROUP,
        group_id=7,
        client_id=9,
    )
    if failure:
        with pytest.raises(ConnectionError, match=f"{failure} unavailable"):
            await workflow.run("docs/", reset=False, scope=scope)
    else:
        await workflow.run("docs/", reset=False, scope=scope)

    assert chunk.metadata["visibility"] == DocumentVisibility.INTERNAL_GROUP.value
    assert chunk.metadata["group_id"] == 7
    assert chunk.metadata["client_id"] == 9
    assert chunk.metadata["doc_domain"] == "general"
    assert events == (["sync"] if failure else ["sync", "registry"])
    if failure:
        registry.persist_entries.assert_not_awaited()
    else:
        registry.persist_entries.assert_awaited_once_with(registry.collect_entries.return_value)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_fails", [False, True])
async def test_single_file_workflow_syncs_before_recording_registry_entry(sync_fails):
    file_info = SimpleNamespace(
        key="docs/report.pdf",
        filename="report.pdf",
        extension=".pdf",
        size_bytes=42,
        last_modified=datetime(2026, 1, 1),
    )
    document = Document(page_content="report text", metadata={})
    chunk = Document(page_content="report text", metadata={})
    events: list[str] = []

    file_storage = MagicMock()
    file_storage.get_file_info = AsyncMock(return_value=file_info)
    file_storage.supported_extensions = (".pdf",)
    loader = MagicMock()
    loader.load_file = AsyncMock(return_value=[document])
    loader.classify_text_domain.return_value = "general"
    loader.index_docs.return_value = [chunk]
    loader.s3_source_key.return_value = "s3://bucket/docs/report.pdf"
    registry = MagicMock()
    registry.list_all = AsyncMock(return_value={})
    registry.is_indexed = AsyncMock(return_value=False)

    async def upsert(*args, **kwargs):
        events.append("registry")

    registry.upsert = AsyncMock(side_effect=upsert)
    sync = MagicMock()

    async def sync_documents(*args, **kwargs):
        events.append("sync")
        if sync_fails:
            raise ConnectionError("sync unavailable")

    sync.sync_documents_to_db = AsyncMock(side_effect=sync_documents)

    workflow = SingleFileIngestionWorkflow(file_storage, loader, registry, sync)
    action = workflow.run(
        "docs/report.pdf",
        scope=IngestionScope(visibility=DocumentVisibility.INTERNAL_GROUP, group_id=7, client_id=9),
    )
    if sync_fails:
        with pytest.raises(ConnectionError, match="sync unavailable"):
            await action
    else:
        await action

    assert chunk.metadata["visibility"] == DocumentVisibility.INTERNAL_GROUP.value
    assert chunk.metadata["group_id"] == 7
    assert chunk.metadata["client_id"] == 9
    assert events == (["sync"] if sync_fails else ["sync", "registry"])
    if sync_fails:
        registry.upsert.assert_not_awaited()
    else:
        registry.upsert.assert_awaited_once_with(
            "docs/report.pdf", "42_2026-01-01 00:00:00", "s3://bucket/docs/report.pdf", 1, chars=11
        )
