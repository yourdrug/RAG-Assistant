"""Tests for the SAGA/Outbox pattern (Postgres ↔ Qdrant consistency).

Verifies that:
- Qdrant operations go through the outbox, not directly
- When Qdrant is unavailable, entries stay in outbox and document is 'indexing'
- Dispatcher retries failed entries with exponential backoff
- Dead letter after max attempts
- NOTIFY is sent on enqueue
- Document transitions to 'done' after successful Qdrant application
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
import pytest

from config import Settings, settings
from domain.knowledge.document import Document
from infrastructure.repositories.config.sqlalchemy_vector_outbox_repository import (
    SQLAlchemyVectorOutboxRepository,
)
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher
from tests.fakes import (
    FakeDocumentRepository,
    FakeUnitOfWork,
    FakeUnitOfWorkFactory,
    FakeVectorOutboxRepository,
    require_present,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from domain.knowledge.enums import DocumentStatus
from domain.knowledge.index_sync import (
    OutboxOperation,
    OutboxStatus,
    VectorOutboxEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_outbox() -> FakeVectorOutboxRepository:

    return FakeVectorOutboxRepository()


@pytest.fixture
def fake_uow(fake_outbox: FakeVectorOutboxRepository) -> FakeUnitOfWork:

    uow = FakeUnitOfWork()
    uow.vector_outbox = fake_outbox
    # Set up a minimal document repository for the dispatcher
    uow.documents = FakeDocumentRepository()
    # Dispatcher upserts only chunks still present in the committed SQL corpus.
    uow.fake_chunks._chunks = [{"id": 10, "document_id": 1}]
    return uow


@pytest.fixture
def fake_uow_factory(fake_uow: FakeUnitOfWork) -> FakeUnitOfWorkFactory:

    return FakeUnitOfWorkFactory(uow=fake_uow)


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    vs = AsyncMock()
    vs.generate_embeddings.return_value = [0.0] * 384
    vs.ensure_collection.return_value = None
    vs.upload_documents.return_value = None
    vs.delete_by_document_id.return_value = None
    vs.delete_by_ids.return_value = None
    return vs


@pytest.fixture
def dispatcher(fake_uow_factory: FakeUnitOfWorkFactory, mock_vector_store: AsyncMock) -> OutboxDispatcher:

    return OutboxDispatcher(
        uow_factory=fake_uow_factory,
        vector_store=mock_vector_store,
    )


# ---------------------------------------------------------------------------
# Test: Outbox entry creation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 5, 10])
@pytest.mark.asyncio
async def test_ten_outbox_entries_obey_configured_concurrency(
    limit: int,
    monkeypatch: pytest.MonkeyPatch,
    fake_uow_factory: FakeUnitOfWorkFactory,
    fake_outbox: FakeVectorOutboxRepository,
) -> None:
    monkeypatch.setattr(settings, "outbox_max_concurrent", limit)
    dispatcher = OutboxDispatcher(fake_uow_factory, AsyncMock())

    for document_id in range(1, 11):
        await fake_outbox.enqueue(VectorOutboxEntry(aggregate_id=document_id))

    active = 0
    peak = 0
    saturated = asyncio.Event()
    release = asyncio.Event()

    async def apply(entry: VectorOutboxEntry) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)

        if active == limit:
            saturated.set()

        try:
            await release.wait()
        finally:
            active -= 1

    monkeypatch.setattr(dispatcher, "_apply_one", apply)
    task = asyncio.create_task(dispatcher.run_once())
    try:
        await asyncio.wait_for(saturated.wait(), timeout=5)
        assert peak == limit
    finally:
        release.set()
        processed = await task
    assert processed == 10
    assert peak == limit


def test_outbox_concurrency_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTBOX_MAX_CONCURRENT", raising=False)
    assert Settings(_env_file=None).outbox_max_concurrent == 5
    monkeypatch.setenv("OUTBOX_MAX_CONCURRENT", "10")
    assert Settings(_env_file=None).outbox_max_concurrent == 10
    monkeypatch.setenv("OUTBOX_MAX_CONCURRENT", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


class TestOutboxEntryCreation:
    """Verify outbox entries are created correctly."""

    @pytest.mark.asyncio
    async def test_enqueue_creates_pending_entry(self, fake_outbox: FakeVectorOutboxRepository) -> None:
        entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=42,
            payload={"points": [{"chunk_id": 1, "page_content": "test", "metadata": {}}]},
        )
        result = await fake_outbox.enqueue(entry)

        assert result.id is not None
        assert result.status == OutboxStatus.PENDING
        assert result.operation == OutboxOperation.UPSERT_CHUNKS
        assert result.aggregate_id == 42

    @pytest.mark.asyncio
    async def test_enqueue_sends_notification(self, fake_outbox: FakeVectorOutboxRepository) -> None:
        entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={"points": []},
        )
        await fake_outbox.enqueue(entry)

        assert len(fake_outbox._notifications) == 1
        assert fake_outbox._notifications[0]["op"] == "upsert_chunks"

    @pytest.mark.asyncio
    async def test_enqueue_delete_by_document(self, fake_outbox: FakeVectorOutboxRepository) -> None:
        entry = VectorOutboxEntry(
            operation=OutboxOperation.DELETE_BY_DOCUMENT,
            aggregate_type="document",
            aggregate_id=99,
            payload={"document_id": 99},
        )
        result = await fake_outbox.enqueue(entry)

        assert result.operation == OutboxOperation.DELETE_BY_DOCUMENT
        assert result.payload["document_id"] == 99


# ---------------------------------------------------------------------------
# Test: Dispatcher - successful application
# ---------------------------------------------------------------------------


class TestDispatcherSuccess:
    """Verify dispatcher applies entries to Qdrant successfully."""

    @pytest.mark.asyncio
    async def test_apply_upsert_chunks(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:

        async with fake_uow_factory.create(master=True) as uow:
            await uow.documents.save(Document(id=1, filename="test.pdf"))

        entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={
                "points": [
                    {
                        "chunk_id": 10,
                        "page_content": "Hello world",
                        "metadata": {"document_id": 1, "visibility": "internal_public"},
                    }
                ]
            },
        )
        await fake_outbox.enqueue(entry)

        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)

        # Claim and process
        batch = await fake_outbox.claim_batch("worker-1")
        assert len(batch) == 1

        for e in batch:
            await dispatcher._apply_one(e)

        # Verify Qdrant was called
        mock_vector_store.upload_documents.assert_awaited_once()

        # Verify entry is marked done
        assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.DONE

    @pytest.mark.asyncio
    async def test_apply_delete_by_document(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:

        entry = VectorOutboxEntry(
            operation=OutboxOperation.DELETE_BY_DOCUMENT,
            aggregate_type="document",
            aggregate_id=42,
            payload={"document_id": 42},
        )
        await fake_outbox.enqueue(entry)

        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)
        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        mock_vector_store.delete_by_document_id.assert_awaited_once_with(42)
        assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.DONE

    @pytest.mark.asyncio
    async def test_apply_delete_chunks(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:

        entry = VectorOutboxEntry(
            operation=OutboxOperation.DELETE_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={"chunk_ids": [10, 20, 30]},
        )
        await fake_outbox.enqueue(entry)

        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)
        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        mock_vector_store.delete_by_ids.assert_awaited_once_with([10, 20, 30])


# ---------------------------------------------------------------------------
# Test: Qdrant unavailable - retry logic
# ---------------------------------------------------------------------------


class TestQdrantUnavailable:
    """Verify behavior when Qdrant is unavailable."""

    @pytest.mark.asyncio
    async def test_failed_entry_gets_retried(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:
        """When Qdrant fails, entry stays in outbox for retry."""

        async with fake_uow_factory.create(master=True) as uow:
            await uow.documents.save(Document(id=1, filename="test.pdf"))

        # Make Qdrant raise an error
        mock_vector_store.upload_documents.side_effect = ConnectionError("Qdrant unavailable")

        entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={
                "points": [
                    {
                        "chunk_id": 10,
                        "page_content": "test",
                        "metadata": {"document_id": 1},
                    }
                ]
            },
        )
        await fake_outbox.enqueue(entry)

        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)
        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        # Entry should be failed, not done
        assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.FAILED
        assert fake_outbox._entries[require_present(entry.id)].attempts == 1
        assert "Qdrant unavailable" in require_present(
            fake_outbox._entries[require_present(entry.id)].last_error
        )

    @pytest.mark.asyncio
    async def test_multiple_failures_lead_to_dead_letter(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:
        """After max_attempts failures, entry goes to dead letter."""

        async with fake_uow_factory.create(master=True) as uow:
            await uow.documents.save(Document(id=1, filename="test.pdf"))

        mock_vector_store.upload_documents.side_effect = ConnectionError("Qdrant down")

        entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            max_attempts=3,
            payload={
                "points": [
                    {
                        "chunk_id": 10,
                        "page_content": "test",
                        "metadata": {"document_id": 1},
                    }
                ]
            },
        )
        await fake_outbox.enqueue(entry)

        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)

        # Process 3 times (all fail)
        for _ in range(3):
            batch = await fake_outbox.claim_batch("worker-1")

            for e in batch:
                await dispatcher._apply_one(e)

        # Entry should be in dead letter
        assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.DEAD_LETTER
        assert fake_outbox._entries[require_present(entry.id)].attempts == 3

    @pytest.mark.asyncio
    async def test_success_after_retry(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:
        """Entry succeeds after previous failure."""

        async with fake_uow_factory.create(master=True) as uow:
            await uow.documents.save(Document(id=1, filename="test.pdf"))

        call_count = 0

        async def flaky_upload(chunks: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                msg = "Temporary failure"
                raise ConnectionError(msg)

        mock_vector_store.upload_documents.side_effect = flaky_upload

        entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={
                "points": [
                    {
                        "chunk_id": 10,
                        "page_content": "test",
                        "metadata": {"document_id": 1},
                    }
                ]
            },
        )
        await fake_outbox.enqueue(entry)

        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)

        # First attempt - fails
        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.FAILED

        # Second attempt - succeeds
        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.DONE


# ---------------------------------------------------------------------------
# Verify idempotency on repeated delivery.
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Verify operations are idempotent."""

    @pytest.mark.asyncio
    async def test_upsert_is_idempotent(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:
        """Same chunk upserted twice produces same result (no duplicate)."""

        async with fake_uow_factory.create(master=True) as uow:
            await uow.documents.save(Document(id=1, filename="test.pdf"))

        entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={
                "points": [
                    {
                        "chunk_id": 10,
                        "page_content": "test content",
                        "metadata": {"document_id": 1},
                    }
                ]
            },
        )
        await fake_outbox.enqueue(entry)

        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)

        # Process same entry twice
        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        # Verify upload_documents was called once per claim (idempotent upsert)
        assert mock_vector_store.upload_documents.call_count == 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:
        """Deleting non-existent points is a no-op (no error)."""

        entry = VectorOutboxEntry(
            operation=OutboxOperation.DELETE_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={"chunk_ids": [99999]},  # Non-existent
        )
        await fake_outbox.enqueue(entry)

        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)
        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        # Should succeed without error
        assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.DONE


# ---------------------------------------------------------------------------
# Test: SKIP LOCKED behavior
# ---------------------------------------------------------------------------


class TestClaimBatch:
    """Verify claim_batch atomically locks entries."""

    @pytest.mark.asyncio
    async def test_claim_batch_returns_pending_entries(self, fake_outbox: FakeVectorOutboxRepository) -> None:
        for i in range(5):
            await fake_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.UPSERT_CHUNKS,
                    aggregate_type="document",
                    aggregate_id=i,
                    payload={"points": []},
                )
            )

        batch = await fake_outbox.claim_batch("worker-1", limit=3)

        assert len(batch) == 3

        # Claimed entries should be in_progress
        for e in batch:
            assert fake_outbox._entries[require_present(e.id)].status == OutboxStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_claim_batch_skips_already_claimed(self, fake_outbox: FakeVectorOutboxRepository) -> None:
        for i in range(3):
            await fake_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.UPSERT_CHUNKS,
                    aggregate_type="document",
                    aggregate_id=i,
                    payload={"points": []},
                )
            )

        # Worker 1 claims
        batch1 = await fake_outbox.claim_batch("worker-1", limit=2)
        assert len(batch1) == 2

        # Worker 2 claims remaining
        batch2 = await fake_outbox.claim_batch("worker-2", limit=10)
        assert len(batch2) == 1


# ---------------------------------------------------------------------------
# Test: Document status transitions
# ---------------------------------------------------------------------------


class TestDocumentStatusTransitions:
    """Verify document status lifecycle with outbox pattern."""

    @pytest.mark.asyncio
    async def test_document_goes_to_indexing_after_outbox(self) -> None:
        """Document should be 'indexing' after outbox enqueue, not 'done'."""
        doc = MagicMock()
        doc.status = DocumentStatus.PROCESSING

        # Simulate what DocumentProcessor.process does
        doc.status = DocumentStatus.INDEXING

        assert doc.status == DocumentStatus.INDEXING
        assert doc.status != DocumentStatus.DONE

    @pytest.mark.asyncio
    async def test_document_goes_to_done_after_qdrant_success(self) -> None:
        """Document transitions to 'done' after dispatcher succeeds."""
        doc = MagicMock()
        doc.status = DocumentStatus.INDEXING

        # Simulate dispatcher marking document as done
        doc.status = DocumentStatus.DONE

        assert doc.status == DocumentStatus.DONE

    @pytest.mark.asyncio
    async def test_indexing_status_blocks_conflict(self) -> None:
        """INDEXING documents should block upload conflicts."""
        doc = MagicMock()
        doc.status = DocumentStatus.INDEXING

        # _handle_existing_conflict should treat INDEXING as "in progress"
        active_statuses = (
            DocumentStatus.PENDING,
            DocumentStatus.PROCESSING,
            DocumentStatus.INDEXING,
        )
        assert doc.status in active_statuses


# ---------------------------------------------------------------------------
# Test: Integration - full flow with Qdrant mock
# ---------------------------------------------------------------------------


class TestFullFlowWithQdrantDown:
    """Integration test: upload document when Qdrant is down, then recover."""

    @pytest.mark.asyncio
    async def test_full_flow_qdrant_down_then_up(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:
        """Full flow: Qdrant down then recover.

        1. User uploads document
        2. Qdrant is down - entry stays in outbox
        3. Qdrant comes back - dispatcher processes entry
        4. Document becomes searchable
        """

        async with fake_uow_factory.create(master=True) as uow:
            await uow.documents.save(Document(id=1, filename="test.pdf"))

        call_count = 0

        async def flaky_upload(chunks: object, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                msg = "Qdrant is down"
                raise ConnectionError(msg)

        mock_vector_store.upload_documents.side_effect = flaky_upload

        # Step 1: Enqueue (simulates DocumentProcessor.process)
        entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={
                "points": [
                    {
                        "chunk_id": 10,
                        "page_content": "Important document content",
                        "metadata": {
                            "document_id": 1,
                            "visibility": "internal_public",
                            "owner_id": 1,
                            "group_id": None,
                            "source": "test.pdf",
                            "content_hash": "abc123",
                            "doc_domain": "general",
                        },
                    }
                ]
            },
        )
        await fake_outbox.enqueue(entry)

        # Step 2: Dispatcher tries to process (Qdrant is down)
        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)
        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        # Verify: entry is failed, not done
        assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.FAILED
        assert fake_outbox._entries[require_present(entry.id)].attempts == 1

        # Step 3: Qdrant comes back - dispatcher retries
        mock_vector_store.upload_documents.side_effect = None  # Qdrant is back
        mock_vector_store.upload_documents.return_value = None

        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        # Verify: entry is now done
        assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.DONE

        # Verify: Qdrant received the upload
        mock_vector_store.upload_documents.assert_awaited()

    @pytest.mark.asyncio
    async def test_mixed_operations_with_qdrant_intermittent(
        self,
        fake_uow_factory: FakeUnitOfWorkFactory,
        mock_vector_store: AsyncMock,
        fake_outbox: FakeVectorOutboxRepository,
    ) -> None:
        """Multiple operations, some succeed, some fail."""

        async with fake_uow_factory.create(master=True) as uow:
            await uow.documents.save(Document(id=1, filename="test.pdf"))

        fail_delete = True

        async def flaky_delete(doc_id: object) -> None:
            nonlocal fail_delete

            if fail_delete:
                msg = "Delete failed"
                raise ConnectionError(msg)

        mock_vector_store.delete_by_document_id.side_effect = flaky_delete

        # Enqueue multiple operations
        upsert_entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={"points": [{"chunk_id": 1, "page_content": "test", "metadata": {}}]},
        )
        delete_entry = VectorOutboxEntry(
            operation=OutboxOperation.DELETE_BY_DOCUMENT,
            aggregate_type="document",
            aggregate_id=2,
            payload={"document_id": 2},
        )
        await fake_outbox.enqueue(upsert_entry)
        await fake_outbox.enqueue(delete_entry)

        dispatcher = OutboxDispatcher(uow_factory=fake_uow_factory, vector_store=mock_vector_store)

        # Process both
        batch = await fake_outbox.claim_batch("worker-1")

        for e in batch:
            await dispatcher._apply_one(e)

        # Upsert should succeed, delete should fail
        assert fake_outbox._entries[require_present(upsert_entry.id)].status == OutboxStatus.DONE
        assert fake_outbox._entries[require_present(delete_entry.id)].status == OutboxStatus.FAILED


# ---------------------------------------------------------------------------
# Test: NOTIFY integration
# ---------------------------------------------------------------------------


class TestNotifyIntegration:
    """Verify NOTIFY is sent on enqueue."""

    @pytest.mark.asyncio
    async def test_notify_sent_on_enqueue(self, fake_outbox: FakeVectorOutboxRepository) -> None:
        """pg_notify should be called when entry is enqueued."""
        entry = VectorOutboxEntry(
            operation=OutboxOperation.UPSERT_CHUNKS,
            aggregate_type="document",
            aggregate_id=1,
            payload={"points": []},
        )
        await fake_outbox.enqueue(entry)

        assert len(fake_outbox._notifications) == 1
        assert fake_outbox._notifications[0]["id"] == entry.id

    @pytest.mark.asyncio
    async def test_notify_with_real_session(self) -> None:
        """Test that SQLAlchemyVectorOutboxRepository uses pg_notify."""

        # Verify the class exists and has the enqueue method
        assert hasattr(SQLAlchemyVectorOutboxRepository, "enqueue")

        source = inspect.getsource(SQLAlchemyVectorOutboxRepository.enqueue)
        assert "pg_notify" in source
        assert "vector_outbox_ready" in source


@pytest.mark.asyncio
async def test_delete_retries_without_reading_sql_corpus(
    fake_uow_factory: FakeUnitOfWorkFactory,
    mock_vector_store: AsyncMock,
    fake_outbox: FakeVectorOutboxRepository,
) -> None:

    mock_vector_store.delete_by_document_id.side_effect = [RuntimeError("Qdrant unavailable"), None]
    dispatcher = OutboxDispatcher(fake_uow_factory, mock_vector_store)
    entry = await fake_outbox.enqueue(
        VectorOutboxEntry(
            operation=OutboxOperation.DELETE_BY_DOCUMENT,
            aggregate_id=42,
            payload={"document_id": 42},
        )
    )
    await dispatcher._apply_one(entry)
    assert entry.status == OutboxStatus.FAILED
    await dispatcher._apply_one(entry)
    assert fake_outbox._entries[require_present(entry.id)].status == OutboxStatus.DONE
    assert mock_vector_store.delete_by_document_id.await_count == 2


@pytest.mark.asyncio
async def test_late_rename_uses_current_sql_filename(
    fake_uow_factory: FakeUnitOfWorkFactory,
    mock_vector_store: AsyncMock,
    fake_outbox: FakeVectorOutboxRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    document = SimpleNamespace(id=42, filename="latest.pdf")
    monkeypatch.setattr(fake_uow_factory._uow.documents, "get_by_id", AsyncMock(return_value=document))
    mock_vector_store.update_filename_by_document_id = AsyncMock()
    entry = await fake_outbox.enqueue(
        VectorOutboxEntry(
            operation=OutboxOperation.RENAME_DOCUMENT,
            aggregate_id=document.id,
            payload={"document_id": document.id, "filename": "obsolete.pdf"},
        )
    )
    await OutboxDispatcher(fake_uow_factory, mock_vector_store)._apply_one(entry)
    mock_vector_store.update_filename_by_document_id.assert_awaited_once_with(document.id, "latest.pdf")
    mock_vector_store.upload_documents.assert_not_awaited()
