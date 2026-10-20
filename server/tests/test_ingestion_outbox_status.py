"""CLI ingestion remains indexing until its vector outbox has been applied."""

from unittest.mock import AsyncMock

import pytest

from application.documents.dto import DocumentDTO
from application.documents.queries import DocumentQueryService
from application.ingestion import DocumentSyncService
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.chunk import ChunkInsert
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import OutboxStatus
from fakes import FakeUnitOfWorkFactory  # type: ignore[import-not-found]
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher


@pytest.mark.asyncio
@pytest.mark.parametrize("reindex", [False, True])
@pytest.mark.parametrize("qdrant_down", [False, True])
async def test_cli_document_waits_for_outbox(
    monkeypatch: pytest.MonkeyPatch, reindex: bool, qdrant_down: bool
) -> None:
    factory = FakeUnitOfWorkFactory()
    uow = factory._uow

    if reindex:
        await uow.documents.save(
            Document(filename="test.md", ingestion_key="test.md", status=DocumentStatus.DONE)
        )
        await uow.chunks.bulk_insert(
            document_id=1,
            filename="test.md",
            visibility=DocumentVisibility.INTERNAL_PUBLIC.value,
            chunks=[ChunkInsert(content="old text")],
        )

    # The shared fake's status methods are no-ops; record persisted status here.
    async def update_status(document_id: int, status: str, **kwargs: object) -> None:
        document = await uow.documents.get_by_id(document_id)
        document.status = DocumentStatus(status)

    monkeypatch.setattr(uow.documents, "update_status", update_status)
    mark_done = AsyncMock(wraps=uow.documents.mark_done_if_indexing)
    monkeypatch.setattr(uow.documents, "mark_done_if_indexing", mark_done)

    await DocumentSyncService(factory).sync_documents_to_db(
        {"test.md": {"hash": "hash", "chunks": 1, "chars": 8, "source": "s3://bucket/test.md"}},
        {},
        [RawDocument("new text", {"source": "s3://bucket/test.md"})],
    )
    document = await uow.documents.get_by_id(1)
    assert document.status == DocumentStatus.INDEXING
    vector_store = AsyncMock()

    if qdrant_down:
        vector_store.upload_documents.side_effect = ConnectionError("Qdrant unavailable")

    dispatcher = OutboxDispatcher(factory, vector_store)

    for _ in range(8 if qdrant_down else 1):
        for entry in await uow.vector_outbox.claim_batch("test-worker"):
            await dispatcher._apply_one(entry)

    if qdrant_down:
        assert any(entry.status == OutboxStatus.DEAD_LETTER for entry in uow.vector_outbox._entries.values())
        mark_done.assert_not_awaited()
        dto = await DocumentQueryService(factory).enrich_with_outbox_status(
            uow, DocumentDTO.from_entity(document)
        )
        assert dto.outbox_pending == 0
        assert dto.outbox_failed == 1
        assert dto.outbox_failed_details is not None
        assert (dto.outbox_failed_details[0]["attempts"], dto.outbox_failed_details[0]["last_error"]) == (
            8,
            "Qdrant unavailable",
        )
    else:
        mark_done.assert_awaited_once_with(document.id)
        assert all(entry.status == OutboxStatus.DONE for entry in uow.vector_outbox._entries.values())
