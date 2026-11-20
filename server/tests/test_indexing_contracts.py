"""Shared indexing exports, transaction ownership and metadata compatibility."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from application import indexing
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import VectorOutboxEntry
from domain.knowledge.revision import REUSE_POINT_KEY, STAGED_REVISION_KEY, StagedRevision


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.indexing", "application.indexing.pipeline", "build_outbox_metadata"),
        ("application.indexing", "application.indexing.pipeline", "classify_domain"),
        ("application.indexing", "application.indexing.pipeline", "enqueue_delete_by_document"),
        ("application.indexing", "application.indexing.pipeline", "enrich_chunks_metadata"),
        ("application.indexing", "application.indexing.pipeline", "process_chunks"),
        ("application.indexing", "application.indexing.pipeline", "process_chunks_in_uow"),
        ("application.indexing", "application.indexing.pipeline", "tag_chunks"),
        ("application.indexing", "application.indexing.pipeline", "tag_domain"),
    ],
)
def test_public_pipeline_exports_share_canonical_objects(public: str, implementation: str, name: str) -> None:
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.indexing", "application.indexing.dto", "DomainClassification"),
        ("application.indexing", "application.indexing.classification", "classify_document_text"),
    ],
)
def test_public_classification_exports_share_canonical_objects(
    public: str, implementation: str, name: str
) -> None:
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )


def contract_uow(events: list[str], published: list[ChunkRecord]) -> Mock:
    uow = Mock()
    uow.chunks.get_ids_by_document_id = AsyncMock(return_value=[chunk.chunk_id for chunk in published])
    uow.chunks.list_document_chunks = AsyncMock(return_value=published)
    uow.chunks.discard_staged_revision = AsyncMock(return_value=[])

    async def insert(**kwargs: object) -> list[int]:
        events.append("insert")
        return [2]

    async def stage(*args: object, **kwargs: object) -> StagedRevision:
        events.append("stage")
        return StagedRevision(9, "token", (2,))

    async def enqueue(entry: VectorOutboxEntry) -> None:
        events.append(entry.operation.value)

    async def update(*args: object, **kwargs: object) -> None:
        events.append("status")

    async def replace_nodes(*args: object) -> None:
        events.append("nodes")

    uow.chunks.bulk_insert = AsyncMock(side_effect=insert)
    uow.chunks.stage_revision = AsyncMock(side_effect=stage)
    uow.chunks.replace_document_nodes = AsyncMock(side_effect=replace_nodes)
    uow.vector_outbox.enqueue = AsyncMock(side_effect=enqueue)
    uow.documents.update_status = AsyncMock(side_effect=update)
    uow.documents.get_by_id = AsyncMock(return_value=SimpleNamespace(space_id=3))
    return uow


def published_chunk(visibility: DocumentVisibility) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=1,
        document_id=9,
        filename="file.txt",
        content="text",
        chunk_index=0,
        visibility=visibility,
        owner_id=7,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("set_indexing", [False, True])
@pytest.mark.parametrize(
    "published,expected_events",
    [
        ([], ["insert", "nodes", "upsert_chunks", "status"]),
        # A changed ACL replaces the published chunks at once.
        (
            [published_chunk(DocumentVisibility.INTERNAL_PUBLIC)],
            ["insert", "nodes", "delete_chunks", "upsert_chunks", "status"],
        ),
        # Same ACL: the new chunks wait as a staged revision; nothing is deleted yet.
        ([published_chunk(DocumentVisibility.INTERNAL_PRIVATE)], ["stage", "upsert_chunks", "status"]),
    ],
    ids=["first", "acl_changed", "staged"],
)
async def test_existing_uow_keeps_transaction_ownership_and_projection_order(
    set_indexing: bool, published: list[ChunkRecord], expected_events: list[str]
) -> None:
    events: list[str] = []
    uow = contract_uow(events, published)
    factory = Mock()
    chunks = [RawDocument("text", {"owner_id": 7, "visibility": "internal_private"})]
    await indexing.process_chunks(
        factory,
        9,
        "file.txt",
        chunks,
        DocumentVisibility.INTERNAL_PRIVATE.value,
        owner_id=7,
        group_id=None,
        doc_domain="general",
        set_indexing=set_indexing,
        _existing_uow=uow,
    )

    factory.create.assert_not_called()
    uow.commit.assert_not_called()
    uow.rollback.assert_not_called()
    assert events == expected_events
    uow.chunks.get_ids_by_document_id.assert_awaited_once_with(9, for_update=True)
    upsert = next(
        call.args[0]
        for call in uow.vector_outbox.enqueue.await_args_list
        if call.args[0].operation.value == "upsert_chunks"
    )
    point = upsert.payload["points"][0]
    assert point["metadata"] == {
        "owner_id": 7,
        "visibility": "internal_private",
        "chunk_id": 2,
        "chunk_index": 0,
        # The document's space labels every vector point for the space search condition.
        "space_id": 3,
    }
    staged = expected_events[0] == "stage"
    assert upsert.payload.get(STAGED_REVISION_KEY) == ("token" if staged else None)
    # Unchanged dense input: the staged chunk may reuse the published chunk's vector.
    assert point.get(REUSE_POINT_KEY) == (1 if staged else None)
    uow.documents.update_status.assert_awaited_once_with(
        9,
        status=DocumentStatus.INDEXING.value if set_indexing else DocumentStatus.DONE.value,
        chunks=1,
        chars=4,
    )


def test_document_enrichment_copies_opaque_pack_metadata_and_keeps_acl() -> None:
    chunks = [RawDocument("text", {"owner_id": 99, "domain_metadata": {"point": "chunk"}})]
    indexing.enrich_chunks_metadata(
        chunks,
        1,
        DocumentVisibility.INTERNAL_GROUP.value,
        None,
        7,
        doc_domain="legal",
        domain_metadata={"point": "document", "number": "5"},
        pack_metadata={"edition": 2, "valid_from": "2026-01-02"},
    )
    assert chunks[0].metadata == {
        "document_id": 1,
        "visibility": DocumentVisibility.INTERNAL_GROUP.value,
        "owner_id": None,
        "group_id": 7,
        "doc_domain": "legal",
        "domain_metadata": {"point": "chunk", "number": "5"},
        "pack_metadata": {"edition": 2, "valid_from": "2026-01-02"},
    }


def test_document_without_pack_metadata_carries_no_pack_key() -> None:
    chunks = [RawDocument("text", {})]
    indexing.enrich_chunks_metadata(
        chunks, 1, DocumentVisibility.INTERNAL_PUBLIC.value, None, None, doc_domain="general"
    )
    assert "pack_metadata" not in chunks[0].metadata
