"""Outbox dispatcher — applies pending vector-store operations to Qdrant.

Reads entries from the vector_store_outbox table and applies them to Qdrant.
Idempotent: upsert overwrites by deterministic chunk_id, delete on missing is no-op.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import uuid

from application.ports.cache_invalidator import CacheInvalidatorPort
from config import settings
from domain.entities.chunk import Chunk
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.repositories.vector_store_repository import VectorStoreRepository

log = logging.getLogger("default")

_BASE_BACKOFF_SEC = 5
_MAX_BACKOFF_SEC = 900  # 15 minutes
_OUTBOX_CONCURRENCY = 5  # max parallel entry processing tasks


class OutboxDispatcher:
    """Applies pending outbox entries to Qdrant.

    Idempotent by construction: upsert by deterministic point_id (chunk.id),
    delete operations are no-op on non-existent points.  Therefore at-least-once
    delivery (retries after mid-flight failure) is safe.
    """

    def __init__(
        self,
        uow_factory,
        vector_store: VectorStoreRepository,
        cache_invalidator: CacheInvalidatorPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store
        self._cache_invalidator = cache_invalidator
        self._worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._semaphore = asyncio.Semaphore(_OUTBOX_CONCURRENCY)

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

        async def _guarded(entry: VectorOutboxEntry) -> None:
            async with self._semaphore:
                await self._apply_one(entry)

        await asyncio.gather(*[_guarded(entry) for entry in batch])

        return len(batch)

    async def _recover_stuck(self) -> int:
        """Reset in_progress entries stuck for >5 minutes (crash recovery).

        These entries were claimed by a dispatcher that crashed before mark_done/mark_failed.
        We reset them to 'pending' so they'll be retried on the next dispatch cycle.
        """
        try:
            async with self._uow_factory.create(master=True) as uow:
                result = await uow.vector_outbox.recover_stuck(
                    stuck_timeout_minutes=settings.stuck_job_timeout_minutes
                )
                if result:
                    log.warning("Recovered %d stuck in_progress entries (dispatcher crash)", result)
                return result
        except Exception as e:
            log.warning("Failed to recover stuck entries: %s", e)
            return 0

    async def _apply_one(self, entry: VectorOutboxEntry) -> None:
        try:
            if entry.operation == OutboxOperation.UPSERT_CHUNKS:
                if not await self._document_exists(entry.aggregate_id):
                    log.info(
                        "Skipping UPSERT for deleted document %d (entry %d)",
                        entry.aggregate_id,
                        entry.id,
                    )
                else:
                    await self._dispatch(entry)
            else:
                await self._dispatch(entry)
            # Durable retry boundary for cache invalidation, including ordinary
            # chunks, deleted documents and newly uploaded documents.
            if self._cache_invalidator is not None and entry.aggregate_type == "document":
                await self._cache_invalidator.invalidate_by_document_ids(
                    [entry.aggregate_id], raise_on_error=True
                )
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
            await self._apply_upsert(entry.payload, document_id=entry.aggregate_id)
        elif entry.operation == OutboxOperation.DELETE_BY_DOCUMENT:
            await self._vector_store.delete_by_document_id(entry.payload["document_id"])
        elif entry.operation == OutboxOperation.DELETE_CHUNKS:
            await self._vector_store.delete_by_ids(entry.payload["chunk_ids"])
        elif entry.operation == OutboxOperation.UPDATE_METADATA:
            await self._apply_version_metadata(
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

    async def _document_exists(self, document_id: int) -> bool:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            return doc is not None

    async def _apply_version_metadata(self, version_id: int, fallback: dict) -> None:
        """Apply committed state, even when an old outbox entry is retried late."""
        metadata = fallback
        document_ids: list[int] = []
        async with self._uow_factory.create(master=True) as uow:
            version = await uow.act_versions.get_by_id(version_id)
            if version is not None:
                metadata = {
                    "act_id": version.act_id,
                    "is_current": version.is_current,
                    "effective_from": version.effective_from.isoformat() if version.effective_from else None,
                    "effective_to": version.effective_to.isoformat() if version.effective_to else None,
                }
                versions = await uow.act_versions.list_by_act(version.act_id) if version.act_id else [version]
                document_ids = list({item.document_id for item in versions})
        if metadata:
            await self._vector_store.update_metadata_by_act_version(version_id, metadata)
        if self._cache_invalidator is not None and document_ids:
            await self._cache_invalidator.invalidate_by_document_ids(document_ids, raise_on_error=True)

    async def _apply_upsert(self, payload: dict, document_id: int = 0) -> None:
        points = payload["points"]
        chunks = [
            Chunk(
                content=p["page_content"],
                metadata={**p["metadata"], "chunk_id": p["chunk_id"]},
            )
            for p in points
        ]

        async def _doc_cancelled() -> bool:
            return not await self._document_exists(document_id)

        # Ensure collection exists (no-op if it does)
        if chunks:
            await self._vector_store.ensure_collection(settings.embed_dim, reset=False)
        await self._vector_store.upload_documents(chunks, should_cancel=_doc_cancelled)
        # Another ingestion/review may have changed the edition while embedding
        # was running. Do not leave the snapshot in this upsert as the final state.
        version_ids = {chunk.metadata.get("act_version_id") for chunk in chunks}
        for version_id in version_ids:
            if version_id is not None:
                await self._apply_version_metadata(version_id, {})

    async def reconcile_stuck_documents(self) -> int:
        """Find documents stuck in 'indexing' with no pending outbox entries and mark them done.

        Returns the number of documents fixed.
        """
        async with self._uow_factory.create(master=True) as uow:
            fixed_ids = await uow.documents.reconcile_indexing_documents()

        if fixed_ids:
            log.info("Reconciled stuck documents -> done: %s", fixed_ids)

        return len(fixed_ids)
