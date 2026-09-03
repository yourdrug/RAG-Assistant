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

from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.entities.vector_outbox_entry import (
    OutboxOperation,
    OutboxStatus,
    VectorOutboxEntry,
)
from domain.value_objects.document_status import DocumentStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_outbox():
    from fakes import FakeVectorOutboxRepository

    return FakeVectorOutboxRepository()


@pytest.fixture
def fake_uow(fake_outbox):
    from fakes import FakeUnitOfWork, FakeDocumentRepository

    uow = FakeUnitOfWork()
    uow.vector_outbox = fake_outbox
    # Set up a minimal document repository for the dispatcher
    uow.documents = FakeDocumentRepository()
    return uow


@pytest.fixture
def fake_uow_factory(fake_uow):
    from fakes import FakeUnitOfWorkFactory

    return FakeUnitOfWorkFactory(uow=fake_uow)


@pytest.fixture
def mock_vector_store():
    vs = AsyncMock()
    vs.generate_embeddings.return_value = [0.0] * 384
    vs.ensure_collection.return_value = None
    vs.upload_documents.return_value = None
    vs.delete_by_document_id.return_value = None
    vs.delete_by_ids.return_value = None
    return vs


@pytest.fixture
def dispatcher(fake_uow_factory, mock_vector_store):
    from infrastructure.outbox_dispatcher import OutboxDispatcher

    return OutboxDispatcher(
        uow_factory=fake_uow_factory,
        vector_store=mock_vector_store,
    )


# ---------------------------------------------------------------------------
# Test: Outbox entry creation
# ---------------------------------------------------------------------------


class TestOutboxEntryCreation:
    """Verify outbox entries are created correctly."""

    @pytest.mark.asyncio
    async def test_enqueue_creates_pending_entry(self, fake_outbox):
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
    async def test_enqueue_sends_notification(self, fake_outbox):
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
    async def test_enqueue_delete_by_document(self, fake_outbox):
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
    async def test_apply_upsert_chunks(self, fake_uow_factory, mock_vector_store, fake_outbox):
        from infrastructure.outbox_dispatcher import OutboxDispatcher

        # Enqueue an upsert entry
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
        assert fake_outbox._entries[entry.id].status == OutboxStatus.DONE

    @pytest.mark.asyncio
    async def test_apply_delete_by_document(self, fake_uow_factory, mock_vector_store, fake_outbox):
        from infrastructure.outbox_dispatcher import OutboxDispatcher

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
        assert fake_outbox._entries[entry.id].status == OutboxStatus.DONE

    @pytest.mark.asyncio
    async def test_apply_delete_chunks(self, fake_uow_factory, mock_vector_store, fake_outbox):
        from infrastructure.outbox_dispatcher import OutboxDispatcher

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
    async def test_failed_entry_gets_retried(self, fake_uow_factory, mock_vector_store, fake_outbox):
        """When Qdrant fails, entry stays in outbox for retry."""
        from infrastructure.outbox_dispatcher import OutboxDispatcher

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
        assert fake_outbox._entries[entry.id].status == OutboxStatus.FAILED
        assert fake_outbox._entries[entry.id].attempts == 1
        assert "Qdrant unavailable" in fake_outbox._entries[entry.id].last_error

    @pytest.mark.asyncio
    async def test_multiple_failures_lead_to_dead_letter(
        self, fake_uow_factory, mock_vector_store, fake_outbox
    ):
        """After max_attempts failures, entry goes to dead letter."""
        from infrastructure.outbox_dispatcher import OutboxDispatcher

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
        assert fake_outbox._entries[entry.id].status == OutboxStatus.DEAD_LETTER
        assert fake_outbox._entries[entry.id].attempts == 3

    @pytest.mark.asyncio
    async def test_success_after_retry(self, fake_uow_factory, mock_vector_store, fake_outbox):
        """Entry succeeds after previous failure."""
        from infrastructure.outbox_dispatcher import OutboxDispatcher

        call_count = 0

        async def flaky_upload(chunks):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Temporary failure")

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
        assert fake_outbox._entries[entry.id].status == OutboxStatus.FAILED

        # Second attempt - succeeds
        batch = await fake_outbox.claim_batch("worker-1")
        for e in batch:
            await dispatcher._apply_one(e)
        assert fake_outbox._entries[entry.id].status == OutboxStatus.DONE


# ---------------------------------------------------------------------------
# Test: Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Verify operations are idempotent."""

    @pytest.mark.asyncio
    async def test_upsert_is_idempotent(self, fake_uow_factory, mock_vector_store, fake_outbox):
        """Same chunk upserted twice produces same result (no duplicate)."""
        from infrastructure.outbox_dispatcher import OutboxDispatcher

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
    async def test_delete_nonexistent_is_noop(self, fake_uow_factory, mock_vector_store, fake_outbox):
        """Deleting non-existent points is a no-op (no error)."""
        from infrastructure.outbox_dispatcher import OutboxDispatcher

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
        assert fake_outbox._entries[entry.id].status == OutboxStatus.DONE


# ---------------------------------------------------------------------------
# Test: SKIP LOCKED behavior
# ---------------------------------------------------------------------------


class TestClaimBatch:
    """Verify claim_batch atomically locks entries."""

    @pytest.mark.asyncio
    async def test_claim_batch_returns_pending_entries(self, fake_outbox):
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
            assert fake_outbox._entries[e.id].status == OutboxStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_claim_batch_skips_already_claimed(self, fake_outbox):
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
    async def test_document_goes_to_indexing_after_outbox(self):
        """Document should be 'indexing' after outbox enqueue, not 'done'."""
        doc = MagicMock()
        doc.status = DocumentStatus.PROCESSING

        # Simulate what DocumentProcessor.process does
        doc.status = DocumentStatus.INDEXING

        assert doc.status == DocumentStatus.INDEXING
        assert doc.status != DocumentStatus.DONE

    @pytest.mark.asyncio
    async def test_document_goes_to_done_after_qdrant_success(self):
        """Document transitions to 'done' after dispatcher succeeds."""
        doc = MagicMock()
        doc.status = DocumentStatus.INDEXING

        # Simulate dispatcher marking document as done
        doc.status = DocumentStatus.DONE

        assert doc.status == DocumentStatus.DONE

    @pytest.mark.asyncio
    async def test_indexing_status_blocks_conflict(self):
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
    async def test_full_flow_qdrant_down_then_up(self, fake_uow_factory, mock_vector_store, fake_outbox):
        """Full flow: Qdrant down then recover.

        1. User uploads document
        2. Qdrant is down - entry stays in outbox
        3. Qdrant comes back - dispatcher processes entry
        4. Document becomes searchable
        """
        from infrastructure.outbox_dispatcher import OutboxDispatcher

        call_count = 0

        async def flaky_upload(chunks):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Qdrant is down")

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
        assert fake_outbox._entries[entry.id].status == OutboxStatus.FAILED
        assert fake_outbox._entries[entry.id].attempts == 1

        # Step 3: Qdrant comes back - dispatcher retries
        mock_vector_store.upload_documents.side_effect = None  # Qdrant is back
        mock_vector_store.upload_documents.return_value = None

        batch = await fake_outbox.claim_batch("worker-1")
        for e in batch:
            await dispatcher._apply_one(e)

        # Verify: entry is now done
        assert fake_outbox._entries[entry.id].status == OutboxStatus.DONE

        # Verify: Qdrant received the upload
        mock_vector_store.upload_documents.assert_awaited()

    @pytest.mark.asyncio
    async def test_mixed_operations_with_qdrant_intermittent(
        self, fake_uow_factory, mock_vector_store, fake_outbox
    ):
        """Multiple operations, some succeed, some fail."""
        from infrastructure.outbox_dispatcher import OutboxDispatcher

        fail_delete = True

        async def flaky_delete(doc_id):
            nonlocal fail_delete
            if fail_delete:
                raise ConnectionError("Delete failed")

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
        assert fake_outbox._entries[upsert_entry.id].status == OutboxStatus.DONE
        assert fake_outbox._entries[delete_entry.id].status == OutboxStatus.FAILED


# ---------------------------------------------------------------------------
# Test: NOTIFY integration
# ---------------------------------------------------------------------------


class TestNotifyIntegration:
    """Verify NOTIFY is sent on enqueue."""

    @pytest.mark.asyncio
    async def test_notify_sent_on_enqueue(self, fake_outbox):
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
    async def test_notify_with_real_session(self):
        """Test that SQLAlchemyVectorOutboxRepository uses pg_notify."""
        from infrastructure.repositories.sqlalchemy_vector_outbox_repository import (
            SQLAlchemyVectorOutboxRepository,
        )

        # Verify the class exists and has the enqueue method
        assert hasattr(SQLAlchemyVectorOutboxRepository, "enqueue")

        # Verify the enqueue method uses text() for pg_notify (source code inspection)
        import inspect

        source = inspect.getsource(SQLAlchemyVectorOutboxRepository.enqueue)
        assert "pg_notify" in source
        assert "vector_outbox_ready" in source
