"""Shared indexing exports, transaction ownership and metadata compatibility."""

import importlib
from unittest.mock import AsyncMock, Mock

import pytest

from application import indexing
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import VectorOutboxEntry


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


@pytest.mark.asyncio
@pytest.mark.parametrize("set_indexing", [False, True])
async def test_existing_uow_keeps_transaction_ownership_and_projection_order(set_indexing: bool) -> None:
    events: list[str] = []
    uow = Mock()
    uow.chunks.get_ids_by_document_id = AsyncMock(return_value=[1])

    async def insert(**kwargs: object) -> list[int]:
        events.append("insert")
        return [2]

    async def enqueue(entry: VectorOutboxEntry) -> None:
        events.append(entry.operation.value)

    async def update(*args: object, **kwargs: object) -> None:
        events.append("status")

    async def replace_nodes(*args: object) -> None:
        events.append("nodes")

    uow.chunks.bulk_insert = AsyncMock(side_effect=insert)
    uow.chunks.replace_document_nodes = AsyncMock(side_effect=replace_nodes)
    uow.vector_outbox.enqueue = AsyncMock(side_effect=enqueue)
    uow.documents.update_status = AsyncMock(side_effect=update)
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
    assert events == ["insert", "nodes", "delete_chunks", "upsert_chunks", "status"]
    uow.chunks.get_ids_by_document_id.assert_awaited_once_with(9, for_update=True)
    points = uow.vector_outbox.enqueue.await_args_list[1].args[0].payload["points"]
    assert points[0]["metadata"] == {
        "owner_id": 7,
        "visibility": "internal_private",
        "chunk_id": 2,
        "chunk_index": 0,
    }
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
