"""Outbox dispatcher — applies pending vector-store operations to Qdrant.

Reads entries from the vector_store_outbox table and applies them to Qdrant.
Idempotent: upsert overwrites by deterministic chunk_id, delete on missing is no-op.
"""

from __future__ import annotations

import logging
import socket
import uuid

from domain.entities.chunk import Chunk
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.repositories.vector_store_repository import VectorStoreRepository

log = logging.getLogger("default")

_BASE_BACKOFF_SEC = 5
_MAX_BACKOFF_SEC = 900  # 15 minutes
_STUCK_TIMEOUT_MINUTES = 30


class OutboxDispatcher:
    """Applies pending outbox entries to Qdrant.

    Idempotent by construction: upsert by deterministic point_id (chunk.id),
    delete operations are no-op on non-existent points.  Therefore at-least-once
    delivery (retries after mid-flight failure) is safe.
    """

    def __init__(self, uow_factory, vector_store: VectorStoreRepository) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store
        self._worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

    async def run_once(self, batch_size: int = 20) -> int:
        """Process one batch of outbox entries. Returns the number of processed entries."""
        # Recover entries stuck in_progress (dispatcher crash recovery)
        recovered = await self._recover_stuck()
        if recovered:
            log.info("Recovered %d stuck in_progress entries", recovered)

        async with self._uow_factory.create(master=True) as uow:
            batch = await uow.vector_outbox.claim_batch(self._worker_id, limit=batch_size)

        if not batch:
            return 0

        for entry in batch:
            await self._apply_one(entry)

        return len(batch)

    async def _recover_stuck(self) -> int:
        """Reset in_progress entries stuck for >5 minutes (crash recovery).

        These entries were claimed by a dispatcher that crashed before mark_done/mark_failed.
        We reset them to 'pending' so they'll be retried on the next dispatch cycle.
        """
        try:
            async with self._uow_factory.create(master=True) as uow:
                result = await uow.vector_outbox.recover_stuck(stuck_timeout_minutes=_STUCK_TIMEOUT_MINUTES)
                if result:
                    log.warning("Recovered %d stuck in_progress entries (dispatcher crash)", result)
                return result
        except Exception as e:
            log.warning("Failed to recover stuck entries: %s", e)
            return 0

    async def _apply_one(self, entry: VectorOutboxEntry) -> None:
        try:
            await self._dispatch(entry)
            async with self._uow_factory.create(master=True) as uow:
                await uow.vector_outbox.mark_done(entry.id)

                # Check if all entries for this document are done
                if entry.aggregate_type == "document":
                    outbox_status = await uow.vector_outbox.count_by_document(entry.aggregate_id)
                    if outbox_status["pending"] == 0:
                        transitioned = await uow.documents.mark_done_if_indexing(entry.aggregate_id)
                        if transitioned:
                            log.info(
                                "Document %d marked as done (all outbox entries applied)",
                                entry.aggregate_id,
                            )

            log.info(
                "Outbox entry %d (%s, aggregate=%s:%d) applied successfully",
                entry.id,
                entry.operation,
                entry.aggregate_type,
                entry.aggregate_id,
            )
        except Exception as e:
            log.warning(
                "Outbox entry %d (%s, aggregate=%s:%d) failed (attempt %d): %s",
                entry.id,
                entry.operation,
                entry.aggregate_type,
                entry.aggregate_id,
                entry.attempts + 1,
                e,
            )
            backoff = min(_BASE_BACKOFF_SEC * (2**entry.attempts), _MAX_BACKOFF_SEC)
            async with self._uow_factory.create(master=True) as uow:
                await uow.vector_outbox.mark_failed(entry.id, str(e), backoff_seconds=backoff)

    async def _dispatch(self, entry: VectorOutboxEntry) -> None:
        if entry.operation == OutboxOperation.UPSERT_CHUNKS:
            await self._apply_upsert(entry.payload)
        elif entry.operation == OutboxOperation.DELETE_BY_DOCUMENT:
            await self._vector_store.delete_by_document_id(entry.payload["document_id"])
        elif entry.operation == OutboxOperation.DELETE_CHUNKS:
            await self._vector_store.delete_by_ids(entry.payload["chunk_ids"])
        elif entry.operation == OutboxOperation.UPDATE_METADATA:
            await self._vector_store.update_metadata_by_act_version(
                entry.payload["act_version_id"],
                {k: v for k, v in entry.payload.items() if k != "act_version_id"},
            )
        elif entry.operation == OutboxOperation.SET_DOCUMENT_ID:
            await self._vector_store.set_document_id_by_source(
                entry.payload["source"],
                entry.payload["document_id"],
            )
        else:
            raise ValueError(f"Unknown outbox operation: {entry.operation}")

    async def _apply_upsert(self, payload: dict) -> None:
        points = payload["points"]
        chunks = [
            Chunk(
                content=p["page_content"],
                metadata={**p["metadata"], "chunk_id": p["chunk_id"]},
            )
            for p in points
        ]
        # Ensure collection exists (no-op if it does)
        if chunks:
            from config import settings

            await self._vector_store.ensure_collection(settings.embed_dim, reset=False)
        await self._vector_store.upload_documents(chunks)

    async def reconcile_stuck_documents(self) -> int:
        """Find documents stuck in 'indexing' with no pending outbox entries and mark them done.

        Returns the number of documents fixed.
        """
        async with self._uow_factory.create(master=True) as uow:
            fixed_ids = await uow.documents.reconcile_indexing_documents()

        if fixed_ids:
            log.info("Reconciled stuck documents -> done: %s", fixed_ids)

        return len(fixed_ids)
