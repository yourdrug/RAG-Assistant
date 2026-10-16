"""Shared indexing exports, transaction ownership and metadata compatibility."""

from datetime import UTC, date, datetime
import importlib
from unittest.mock import AsyncMock, Mock

import pytest

from application import indexing
from application.indexing.pipeline import parse_meta_datetime
from domain.entities.raw_document import RawDocument
from domain.entities.vector_outbox_entry import VectorOutboxEntry
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.visibility import DocumentVisibility


@pytest.mark.parametrize(
    "name",
    [
        "build_outbox_metadata",
        "classify_domain",
        "enqueue_delete_by_document",
        "enrich_chunks_metadata",
        "process_chunks",
        "process_chunks_in_uow",
        "tag_chunks",
        "tag_domain",
    ],
)
def test_legacy_pipeline_exports_share_canonical_objects(name: str) -> None:
    assert getattr(importlib.import_module("application.services.document_pipeline"), name) is getattr(
        indexing, name
    )


@pytest.mark.parametrize("name", ["DomainClassification", "classify_document_text"])
def test_legacy_classification_exports_share_canonical_objects(name: str) -> None:
    assert getattr(importlib.import_module("application.services.domain_classification"), name) is getattr(
        indexing, name
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (date(2026, 1, 2), date(2026, 1, 2)),
        (datetime(2026, 1, 2, 12, tzinfo=UTC), date(2026, 1, 2)),
        ("2026-01-02", date(2026, 1, 2)),
        ("2026-01-02T12:30:00+03:00", date(2026, 1, 2)),
        ("invalid", None),
    ],
)
def test_temporal_metadata_normalizes_to_date(value: object, expected: date | None) -> None:
    assert parse_meta_datetime(value) == expected


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

    uow.chunks.bulk_insert = AsyncMock(side_effect=insert)
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
    assert events == ["insert", "delete_chunks", "upsert_chunks", "status"]
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
        DocumentStatus.INDEXING.value if set_indexing else DocumentStatus.DONE.value,
        chunks=1,
        chars=4,
    )


def test_version_enrichment_preserves_chunk_references_and_acl() -> None:
    chunks = [RawDocument("text", {"owner_id": 99, "domain_metadata": {"point": "chunk"}})]
    indexing.enrich_chunks_metadata(
        chunks,
        1,
        DocumentVisibility.INTERNAL_GROUP.value,
        None,
        7,
        doc_domain="legal",
        domain_metadata={"point": "document", "number": "5"},
        act_version_id=2,
        act_id=3,
        effective_from=date(2026, 1, 2),
        effective_to="2026-02-01",
        is_current=False,
    )
    assert chunks[0].metadata == {
        "document_id": 1,
        "visibility": DocumentVisibility.INTERNAL_GROUP.value,
        "owner_id": None,
        "group_id": 7,
        "doc_domain": "legal",
        "domain_metadata": {"point": "chunk", "number": "5"},
        "act_version_id": 2,
        "act_id": 3,
        "effective_from": "2026-01-02",
        "effective_to": "2026-02-01",
        "is_current": False,
    }
