"""Outbox dispatcher — applies committed operations to Qdrant (dense and sparse).

Reads entries from the vector_store_outbox table and applies them to Qdrant.
Idempotent: upsert overwrites by deterministic chunk_id, delete on missing is no-op.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import TYPE_CHECKING, Any
import uuid

from config import JobsSection, MLProviderSection, process_section
from domain.knowledge.chunk import Chunk
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.index_sync import OutboxOperation, VectorOutboxEntry
from domain.knowledge.pack_metadata import PACK_METADATA_KEY
from domain.knowledge.revision import REUSE_POINT_KEY, STAGED_KEY, STAGED_REVISION_KEY
from domain.shared.hashing import content_hash
from infrastructure.utils.async_tasks import finish_owned_task


if TYPE_CHECKING:
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.uow import UnitOfWork
    from application.uow.factory import UnitOfWorkFactory
    from domain.knowledge.chunk import ChunkRecord
    from domain.knowledge.index_sync import VectorIndexWriter


log = logging.getLogger("default")

_BASE_BACKOFF_SEC = 5
_MAX_BACKOFF_SEC = 900  # 15 minutes
# Chunks read from SQL, embedded and written per step of one document upsert.
_UPSERT_SLICE_SIZE = 256
STAGED_UPDATE_FAILED_WARNING = (
    "Новая версия документа не проиндексирована; поиск использует предыдущую версию"
)


def staged_token(entry: VectorOutboxEntry) -> str | None:
    """Token of the staged revision an upsert entry uploads; ``None`` for published chunks."""
    if entry.operation != OutboxOperation.UPSERT_CHUNKS:
        return None

    token = entry.payload.get(STAGED_REVISION_KEY)
    return token if isinstance(token, str) else None


class OutboxDispatcher:
    """Applies pending outbox entries to Qdrant.

    Upserts and document deletes serialize per document across processes.
    Upserts read committed SQL content and retries converge to current text.
    Deletes are idempotent for missing points.
    """

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store: VectorIndexWriter,
        cache_invalidator: CacheInvalidatorPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store
        self._cache_invalidator = cache_invalidator
        self._host_prefix = f"{socket.gethostname()}-"
        self._worker_id = f"{self._host_prefix}{uuid.uuid4().hex[:8]}"
        self._semaphore = asyncio.Semaphore(process_section(JobsSection).outbox_max_concurrent)

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

    async def recover_abandoned(self) -> int:
        """Requeue entries claimed by an earlier dispatcher process on this host.

        Called once at maintenance startup. One maintenance process runs per host
        (container), and Docker restarts a killed process in the same container,
        so an entry locked under this hostname by another worker id belongs to a
        dead process. Replaying it is safe: every operation is idempotent.
        """
        async with self._uow_factory.create(master=True) as uow:
            result = await uow.vector_outbox.release_abandoned(self._host_prefix, self._worker_id)

        if result:
            log.warning("Requeued %d outbox entries left by a dead dispatcher on this host", result)

        return result

    async def _recover_stuck(self) -> int:
        """Reset in_progress entries locked longer than ``stuck_job_timeout_minutes``.

        These entries were claimed by a dispatcher that crashed before mark_done/mark_failed,
        typically on another host; ``recover_abandoned`` handles this host at startup.
        We reset them to 'pending' so they'll be retried on the next dispatch cycle.
        """
        try:
            async with self._uow_factory.create(master=True) as uow:
                result = await uow.vector_outbox.recover_stuck(
                    stuck_timeout_minutes=process_section(JobsSection).stuck_job_timeout_minutes
                )

                if result:
                    log.warning("Recovered %d stuck in_progress entries (dispatcher crash)", result)

                return result
        except Exception as e:
            log.warning("Failed to recover stuck entries: %s", e, exc_info=True)
            return 0

    async def _apply_one(self, entry: VectorOutboxEntry) -> None:
        if entry.id is None:
            message = "Only persisted outbox entries can be dispatched"
            raise ValueError(message)

        try:
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

                    if outbox_status["pending"] == 0 and outbox_status["failed"] == 0:
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
                exc_info=True,
            )
            await self.record_failure(entry, str(e))

    async def record_failure(self, entry: VectorOutboxEntry, error: str) -> None:
        """Schedule a retry; a dead-lettered upsert leaves the published revision in charge."""
        backoff = min(_BASE_BACKOFF_SEC * (2**entry.attempts), _MAX_BACKOFF_SEC)
        async with self._uow_factory.create(master=True) as uow:
            await uow.vector_outbox.mark_failed(entry.id or 0, error, backoff_seconds=backoff)

            if entry.attempts + 1 < entry.max_attempts or entry.operation != OutboxOperation.UPSERT_CHUNKS:
                return

            token = staged_token(entry)

            if token is not None:
                await self.abandon_staged_revision(uow, entry.aggregate_id, token)
            else:
                await self.forget_source_version(uow, entry.aggregate_id)

    @staticmethod
    async def forget_source_version(uow: UnitOfWork, document_id: int) -> None:
        """A dead-lettered direct upload: the source version is no longer proven indexed.

        A storage-managed document then counts as changed, so its channel retries the file.
        """
        document = await uow.documents.get_by_id(document_id, for_update=True)

        if document is not None and document.sync is not None and document.sync.storage_version is not None:
            await uow.documents.set_storage_version(document_id, None, None)

    async def abandon_staged_revision(self, uow: UnitOfWork, document_id: int, token: str) -> None:
        """A dead-lettered staged upload leaves the published revision in place.

        The staged chunks and any of their uploaded points are removed, and the
        document returns to ``done`` with a warning instead of staying ``indexing``.
        Its source version stays the published one, so a storage channel retries the file.
        """
        # Lock first: a newer processing run may stage a fresh revision meanwhile, and
        # only the revision of this token may be discarded.
        if await uow.documents.get_by_id(document_id, for_update=True) is None:
            return

        staged = await uow.chunks.get_staged_revision(document_id)

        if staged is None or staged.token != token:
            return

        staged_ids = await uow.chunks.discard_staged_revision(document_id)

        if staged_ids:
            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.DELETE_CHUNKS,
                    aggregate_type="document",
                    aggregate_id=document_id,
                    payload={"chunk_ids": staged_ids},
                )
            )

        stats = await uow.chunks.get_document_stats(document_id)
        await uow.documents.update_status(
            document_id,
            status=DocumentStatus.DONE.value,
            chunks=stats.total_chunks,
            chars=stats.total_chars,
            warning=STAGED_UPDATE_FAILED_WARNING,
        )
        log.warning("Staged revision of document %d abandoned after repeated upload failures", document_id)

    async def _dispatch(self, entry: VectorOutboxEntry) -> None:
        operation: str = entry.operation

        if operation == OutboxOperation.UPSERT_CHUNKS:
            await self.apply_upsert(entry.payload, document_id=entry.aggregate_id)
        elif operation == OutboxOperation.DELETE_BY_DOCUMENT:
            # A delete must finish after any in-flight upload and its cleanup.
            # Otherwise the upload can restore points after this entry is DONE.
            document_id = entry.payload["document_id"]
            async with self._uow_factory.document_index_lock(document_id):
                await self._vector_store.delete_by_document_id(document_id)
        elif operation == OutboxOperation.RENAME_DOCUMENT:
            await self.refresh_document_filename(entry.aggregate_id)
        elif operation == OutboxOperation.DELETE_CHUNKS:
            await self._vector_store.delete_by_ids(entry.payload["chunk_ids"])
        elif operation == OutboxOperation.PUBLISH_REVISION:
            # Reveal first: the document never disappears from search between the steps.
            await self._vector_store.reveal_points(entry.payload["published_ids"])
            await self._vector_store.delete_by_ids(entry.payload["replaced_ids"])
        elif operation == OutboxOperation.UPDATE_METADATA:
            await self._apply_pack_metadata(entry.aggregate_id, entry.payload.get("related_document_ids", []))
        elif operation == OutboxOperation.SET_DOCUMENT_ID:
            # Legacy jobs: upserts carry document IDs. A source-wide patch would
            # steal points belonging to another assignment of the same source.
            log.debug("Skipping legacy source-wide document ID patch")
        else:
            msg = f"Unknown outbox operation: {entry.operation}"
            raise ValueError(msg)

    async def refresh_document_filename(self, document_id: int) -> None:
        async with self._uow_factory.create(master=True) as uow:
            document = await uow.documents.get_by_id(document_id)

        if document is not None:
            await self._vector_store.update_filename_by_document_id(document_id, document.filename)

    async def _document_exists(self, document_id: int) -> bool:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            return doc is not None

    async def _apply_pack_metadata(self, document_id: int, related_document_ids: list[int]) -> None:
        """Apply committed state, even when an old outbox entry is retried late."""
        async with self._uow_factory.document_index_lock(document_id):
            await self.project_pack_metadata(document_id)

        # The pack names documents whose answers depend on this change (e.g. other editions).
        others = [item for item in related_document_ids if item != document_id]

        if self._cache_invalidator is not None and others:
            await self._cache_invalidator.invalidate_by_document_ids(others, raise_on_error=True)

    async def project_pack_metadata(self, document_id: int, *, unless_in: set[str] | None = None) -> None:
        """Copy committed SQL pack metadata to the points; the caller holds the document lock.

        ``unless_in`` skips the write when the points already carry exactly this state.
        """
        async with self._uow_factory.create(master=True) as uow:
            pack_metadata = await uow.chunks.get_pack_metadata(document_id)

        if pack_metadata is None:
            return

        if unless_in is not None and unless_in == {json.dumps(pack_metadata, sort_keys=True)}:
            return

        await self._vector_store.set_pack_metadata_by_document_id(document_id, pack_metadata)

    async def current_chunks(
        self, document_id: int, chunk_ids: list[int], token: str | None = None
    ) -> tuple[dict[int, ChunkRecord], set[int]]:
        """Committed chunks and the ids among them that are still staged.

        Without ``token``: published chunks. With it: chunks of that staged revision.
        Once the revision is published its chunks keep their ids, so a redelivered
        upload finds them among the published chunks and must keep their points.
        """
        async with self._uow_factory.create(master=True) as uow:
            if token is None:
                rows = await uow.chunks.get_by_ids(document_id, chunk_ids)
                return {chunk.chunk_id: chunk for chunk in rows}, set()

            staged = {
                chunk.chunk_id: chunk
                for chunk in await uow.chunks.get_staged_by_ids(document_id, token, chunk_ids)
            }
            missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in staged]
            # Ids come from one sequence: a missing staged id found among published chunks
            # means this revision was published already (a redelivery after publication).
            published = await uow.chunks.get_by_ids(document_id, missing) if missing else []
            return {**{chunk.chunk_id: chunk for chunk in published}, **staged}, set(staged)

    async def verify_uploaded_chunks(
        self, document_id: int, chunks: list[Chunk], token: str | None = None
    ) -> None:
        current, _ = await self.current_chunks(
            document_id, [chunk.metadata["chunk_id"] for chunk in chunks], token
        )
        obsolete_ids = [
            chunk.metadata["chunk_id"] for chunk in chunks if chunk.metadata["chunk_id"] not in current
        ]

        if obsolete_ids:
            await self._vector_store.delete_by_ids(obsolete_ids)

        # Concurrent edits keep their IDs; a stale write must retry from SQL.
        if any(
            chunk.metadata["chunk_id"] in current
            and (
                chunk.content != current[chunk.metadata["chunk_id"]].content
                or chunk.embed_text != current[chunk.metadata["chunk_id"]].embed_text
            )
            for chunk in chunks
        ):
            message = "Chunk content changed during vector upload; retry required"
            raise RuntimeError(message)

    async def apply_upsert(self, payload: dict[str, Any], document_id: int = 0) -> None:
        async with self._uow_factory.document_index_lock(document_id):
            # Cancelling an await cannot stop a Qdrant write already running in
            # a thread. Keep the document lock until upload and cleanup finish.
            upload = asyncio.create_task(self.apply_current_upsert(payload, document_id))
            try:
                await asyncio.shield(upload)
            except asyncio.CancelledError as exc:
                try:
                    await finish_owned_task(upload)
                finally:
                    raise exc

    async def apply_current_upsert(self, payload: dict[str, Any], document_id: int) -> None:
        """Upload the document slice by slice, so only one slice of chunks is in memory."""

        async def _doc_cancelled() -> bool:
            return not await self._document_exists(document_id)

        uploaded: set[str] = set()
        points = payload["points"]
        token = payload.get(STAGED_REVISION_KEY)

        for start in range(0, len(points), _UPSERT_SLICE_SIZE):
            chunks = await self.current_slice_chunks(
                points[start : start + _UPSERT_SLICE_SIZE], document_id, token
            )

            if not chunks:
                continue

            if not uploaded:
                # Ensure collection exists (no-op if it does)
                await self._vector_store.ensure_collection(
                    process_section(MLProviderSection).embed_dim, reset=False
                )

            await self._vector_store.upload_documents(chunks, should_cancel=_doc_cancelled)
            await self.verify_uploaded_chunks(document_id, chunks, token)
            uploaded.update(
                json.dumps(chunk.metadata.get(PACK_METADATA_KEY), sort_keys=True) for chunk in chunks
            )

        if token is not None:
            await self.publish_staged_revision(document_id, token)

        if not uploaded:
            return

        # A rename may commit while embeddings are running. Refresh its payload last.
        await self.refresh_document_filename(document_id)
        # A pack may have changed the document metadata while embedding was running.
        # Do not leave the snapshot in this upsert as the final state.
        await self.project_pack_metadata(document_id, unless_in=uploaded)

    async def publish_staged_revision(self, document_id: int, token: str) -> None:
        """Every staged point is indexed: make the revision the one search and citations use."""
        async with self._uow_factory.create(master=True) as uow:
            # Same lock order as re-chunking and chunk edits: the document row first.
            if await uow.documents.get_by_id(document_id, for_update=True) is None:
                return

            switch = await uow.chunks.publish_staged_revision(document_id, token)

            if switch is None:
                log.info("Staged revision of document %d was superseded; not published", document_id)
                return

            if switch.source_version is not None:
                # Only now does the document serve the text of this source version.
                await uow.documents.set_storage_version(
                    document_id, switch.source_version.version, switch.source_version.revision
                )

            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.PUBLISH_REVISION,
                    aggregate_type="document",
                    aggregate_id=document_id,
                    payload={
                        "published_ids": list(switch.published_ids),
                        "replaced_ids": list(switch.replaced_ids),
                    },
                )
            )

        log.info(
            "Document %d: staged revision published (%d chunks, %d replaced)",
            document_id,
            len(switch.published_ids),
            len(switch.replaced_ids),
        )

    async def current_slice_chunks(
        self, points: list[dict[str, Any]], document_id: int, token: str | None = None
    ) -> list[Chunk]:
        """Build chunks from committed SQL content; drop points whose chunks no longer exist."""
        chunk_ids = [p["chunk_id"] for p in points]
        current, staged_ids = await self.current_chunks(document_id, chunk_ids, token)
        obsolete_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in current]

        if obsolete_ids:
            # Also clean up on retries after a failed upload-time deletion.
            await self._vector_store.delete_by_ids(obsolete_ids)

        return [
            Chunk(
                content=current[p["chunk_id"]].content,
                embed_text=current[p["chunk_id"]].embed_text,
                metadata={
                    **p["metadata"],
                    "chunk_id": p["chunk_id"],
                    "content_hash": content_hash(current[p["chunk_id"]].content),
                    PACK_METADATA_KEY: dict(current[p["chunk_id"]].pack_metadata),
                    # A point of an already published revision stays visible on redelivery.
                    **({STAGED_KEY: True} if p["chunk_id"] in staged_ids else {}),
                    **({REUSE_POINT_KEY: p[REUSE_POINT_KEY]} if p.get(REUSE_POINT_KEY) is not None else {}),
                },
            )
            for p in points
            if p["chunk_id"] in current
        ]

    async def reconcile_stuck_documents(self) -> int:
        """Find documents stuck in 'indexing' with no pending outbox entries and mark them done.

        Returns the number of documents fixed.
        """
        async with self._uow_factory.create(master=True) as uow:
            fixed_ids = await uow.documents.reconcile_indexing_documents()

        if fixed_ids:
            log.info("Reconciled stuck documents -> done: %s", fixed_ids)

        return len(fixed_ids)
