"""Re-chunk stored documents with the current pipeline, in resumable batches.

Each document goes through the same processing path as an upload (parse → split →
persist chunks with ``DELETE_CHUNKS`` + ``UPSERT_CHUNKS`` outbox), keeping its id,
ACL and stored domain. Documents with manual chunk edits are skipped and reported:
re-chunking would discard the administrator's edits. The check is repeated under the
document row lock in the persist transaction, so an edit made during processing wins.
A failed run of a DONE document keeps its old chunks and restores status DONE.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import logging
from typing import TYPE_CHECKING, Protocol

from application.documents.dto import ProcessDocumentCommand
from application.documents.persistence import ManualChunkEditsError
from domain.knowledge.enums import DocumentStatus, SourceType


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.uow import UnitOfWorkFactory
    from domain.knowledge.document import Document


log = logging.getLogger("default")

# Pending/processing/indexing documents belong to an in-flight job.
REINDEXABLE_STATUSES = frozenset({DocumentStatus.DONE, DocumentStatus.FAILED})


class DocumentReprocessor(Protocol):
    async def process(self, command: ProcessDocumentCommand) -> None: ...


class ReindexOutcome:
    REINDEXED = "reindexed"
    FAILED = "failed"
    SKIPPED_MANUAL = "skipped_manual"
    SKIPPED_STATUS = "skipped_status"
    SKIPPED_SOURCE = "skipped_source"
    SELECTED = "selected"


@dataclass(frozen=True)
class ReindexRequest:
    doc_domain: str | None = None
    document_ids: tuple[int, ...] = ()
    after_id: int = 0
    limit: int | None = None
    batch_size: int = 10
    pause_seconds: float = 0.0
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            message = "batch_size must be positive"
            raise ValueError(message)

        if self.pause_seconds < 0 or self.after_id < 0 or (self.limit is not None and self.limit < 1):
            message = "pause_seconds and after_id must be non-negative, limit positive"
            raise ValueError(message)


@dataclass(frozen=True)
class ReindexEvent:
    document_id: int
    filename: str
    outcome: str
    detail: str | None = None


@dataclass
class ReindexReport:
    events: list[ReindexEvent] = field(default_factory=list)

    def ids(self, outcome: str) -> list[int]:
        return [event.document_id for event in self.events if event.outcome == outcome]

    @property
    def last_id(self) -> int | None:
        return self.events[-1].document_id if self.events else None


def skip_reason(document: Document) -> tuple[str, str] | None:
    if document.has_manual_edits:
        return ReindexOutcome.SKIPPED_MANUAL, "document has manual chunk edits"

    if document.status not in REINDEXABLE_STATUSES:
        return ReindexOutcome.SKIPPED_STATUS, f"status {document.status.value}"

    if document.source_type != SourceType.FILE.value or not document.source_path:
        return ReindexOutcome.SKIPPED_SOURCE, "no stored source file"

    return None


class DocumentReindexService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        processor: DocumentReprocessor,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._uow_factory = uow_factory
        self._processor = processor
        self._sleep = sleep

    async def run(
        self,
        request: ReindexRequest,
        on_event: Callable[[ReindexEvent], None] | None = None,
    ) -> ReindexReport:
        """Process documents in id order; ``on_event`` sees each one, e.g. to checkpoint."""
        report = ReindexReport()
        after_id = request.after_id
        remaining = request.limit
        first_batch = True

        while remaining is None or remaining > 0:
            size = request.batch_size if remaining is None else min(request.batch_size, remaining)
            batch = await self.next_batch(request, after_id, size)

            if not batch:
                break

            if not first_batch and request.pause_seconds and not request.dry_run:
                await self._sleep(request.pause_seconds)

            first_batch = False

            for document in batch:
                event = await self.reindex_one(document, dry_run=request.dry_run)
                report.events.append(event)

                if on_event is not None:
                    on_event(event)

            after_id = self.document_id(batch[-1])

            if remaining is not None:
                remaining -= len(batch)

        return report

    async def next_batch(self, request: ReindexRequest, after_id: int, size: int) -> list[Document]:
        async with self._uow_factory.create() as uow:
            if request.document_ids:
                selected = sorted(
                    await uow.documents.get_by_ids(list(request.document_ids)), key=self.document_id
                )
                return [
                    doc
                    for doc in selected
                    if self.document_id(doc) > after_id
                    and (request.doc_domain is None or doc.doc_domain == request.doc_domain)
                ][:size]

            return await uow.documents.list_after_id(after_id, size, doc_domain=request.doc_domain)

    async def reindex_one(self, document: Document, *, dry_run: bool) -> ReindexEvent:
        document_id = self.document_id(document)

        # Re-read right before processing: a manual edit or deletion since the
        # batch was listed must not be overwritten by re-chunking.
        async with self._uow_factory.create(master=True) as uow:
            current = await uow.documents.get_by_id(document_id)

        if current is None:
            return ReindexEvent(document_id, document.filename, ReindexOutcome.SKIPPED_STATUS, "deleted")

        if reason := skip_reason(current):
            return ReindexEvent(document_id, current.filename, *reason)

        if dry_run:
            return ReindexEvent(document_id, current.filename, ReindexOutcome.SELECTED, current.doc_domain)

        return await self.process(document_id, current)

    async def process(self, document_id: int, current: Document) -> ReindexEvent:
        try:
            await self._processor.process(
                ProcessDocumentCommand(
                    document_id=document_id,
                    storage_key=current.source_path,
                    original_filename=current.filename,
                    visibility=current.visibility,
                    owner_id=current.owner_id,
                    group_id=current.group_id,
                    replace_id=None,
                    # The stored domain drives ACL-adjacent policy and versions; keep it.
                    doc_domain=current.doc_domain,
                    # A copy: processing must not mutate the state it may restore.
                    reindex_snapshot=replace(current),
                )
            )
        except ManualChunkEditsError:
            # Edited after the re-read: the commit was rolled back, edits are kept.
            return ReindexEvent(
                document_id, current.filename, ReindexOutcome.SKIPPED_MANUAL, "edited during re-chunking"
            )
        except Exception as exc:
            # The processor restored a DONE document or marked it failed; continue the batch.
            log.exception("Reindex failed for document %d", document_id)
            return ReindexEvent(document_id, current.filename, ReindexOutcome.FAILED, type(exc).__name__)

        return ReindexEvent(document_id, current.filename, ReindexOutcome.REINDEXED, current.doc_domain)

    @staticmethod
    def document_id(document: Document) -> int:
        if document.id is None:
            message = "Stored document has no id"
            raise ValueError(message)

        return document.id
