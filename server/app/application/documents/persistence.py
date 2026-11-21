"""Document persistence -- the transactional tail of the processing pipeline.

Single transaction: set_domain → enrich chunks → bulk insert + outbox enqueue →
replacement handling → warning/status update.  Extracted from DocumentProcessor
so the transaction boundary lives in one auditable place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.documents.conflicts import apply_async_replacement, inherited_filename
from application.indexing import enrich_chunks_metadata, process_chunks
from application.indexing.pack_hooks import AppliedPackDocument, apply_pack_document
from domain.knowledge.enums import DocumentStatus
from domain.shared.exceptions import BusinessRuleViolationError


if TYPE_CHECKING:
    from application.documents.dto import PersistDocumentCommand
    from application.indexing.pack_hooks import DocumentPackHooks
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.knowledge.document import Document

log = logging.getLogger("default")


class ManualChunkEditsError(BusinessRuleViolationError):
    """Re-chunking would discard manual chunk edits committed during processing."""


async def lock_document_for_persist(uow: UnitOfWork, command: PersistDocumentCommand) -> Document | None:
    """Lock the row first: chunk edits lock it too, so they cannot interleave with persist.

    A guard runs before: it locks coarser rows (a sync channel), and every path locks
    such a row before the documents under it, so the two orders never deadlock.
    """
    if command.guard is not None:
        await command.guard(uow)

    existing = await uow.documents.get_by_id(command.document_id, for_update=True)

    if existing is not None and command.abort_on_manual_edits and existing.has_manual_edits:
        msg = f"Document {command.document_id} received manual chunk edits during re-chunking"
        raise ManualChunkEditsError(msg)

    return existing


async def persist_document_result(
    uow_factory: UnitOfWorkFactory,
    command: PersistDocumentCommand,
    *,
    domain_registry: DomainProfileRegistry | None,
    pack_hooks: DocumentPackHooks | None,
) -> AppliedPackDocument | None:
    """Commit pack state, chunks, replacement and outbox together, then run pack effects."""
    document_id = command.document_id
    applied = AppliedPackDocument()
    storage_deletes: list[str] = []
    async with uow_factory.create(master=True) as uow:
        current = await lock_document_for_persist(uow, command)

        if current is None:
            log.info("Document %d deleted before outbox enqueue — aborting", document_id)
            return None

        # Chunks and their vector payload carry the final name from the start.
        restored_filename = (
            await inherited_filename(
                uow, domain_registry, command.replace_id, command.doc_domain, current.filename
            )
            if command.replace_id is not None
            else None
        )

        if restored_filename is not None:
            for chunk in command.raw_chunks:
                chunk.metadata["source"] = restored_filename

        applied = await apply_pack_document(pack_hooks, uow, document_id, command.pack_document)

        enrich_chunks_metadata(
            command.raw_chunks,
            document_id,
            command.visibility,
            command.owner_id,
            command.group_id,
            doc_domain=command.doc_domain,
            domain_metadata=applied.domain_metadata,
            pack_metadata=applied.pack_metadata,
        )

        await process_chunks(
            uow_factory=uow_factory,
            document_id=document_id,
            filename=restored_filename or command.original_filename,
            chunks=command.raw_chunks,
            visibility=command.visibility,
            owner_id=command.owner_id,
            group_id=command.group_id,
            doc_domain=command.doc_domain,
            set_indexing=True,
            source_version=command.source_version,
            _existing_uow=uow,
        )

        if command.replace_id is not None:
            await apply_async_replacement(
                uow,
                domain_registry,
                command.replace_id,
                command.doc_domain,
                storage_deletes,
                pack_hooks=pack_hooks,
            )

        if restored_filename is not None:
            # The deleted predecessor freed the name's slot in this transaction.
            await uow.documents.update_filename(document_id, restored_filename, current.source_path)

        if command.warning_message:
            await uow.documents.update_status(
                document_id,
                status=DocumentStatus.INDEXING.value,
                warning=command.warning_message,
                quality_score=command.quality.bad_ratio if command.quality else None,
            )

    # Publish cleanup keys only after commit: a rollback must preserve old files.
    command.storage_deletes.extend(storage_deletes)

    if pack_hooks is not None:
        await pack_hooks.after_commit(document_id, applied)

    return applied
