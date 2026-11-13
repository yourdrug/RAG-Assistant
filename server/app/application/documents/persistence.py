"""Document persistence -- the transactional tail of the processing pipeline.

Single transaction: set_domain → enrich chunks → bulk insert + outbox enqueue →
replacement handling → warning/status update.  Extracted from DocumentProcessor
so the transaction boundary lives in one auditable place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.act_versioning import VersioningResult
from application.documents.conflicts import apply_async_replacement
from application.indexing import enrich_chunks_metadata, process_chunks
from domain.knowledge.enums import DocumentStatus
from domain.shared.exceptions import BusinessRuleViolationError


if TYPE_CHECKING:
    from application.act_versioning import ActVersioningService
    from application.documents.dto import PersistDocumentCommand
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.knowledge.document import Document

log = logging.getLogger("default")


class ManualChunkEditsError(BusinessRuleViolationError):
    """Re-chunking would discard manual chunk edits committed during processing."""


async def lock_document_for_persist(uow: UnitOfWork, command: PersistDocumentCommand) -> Document | None:
    """Lock the row first: chunk edits lock it too, so they cannot interleave with persist."""
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
    act_versioning_service: ActVersioningService | None,
) -> VersioningResult | None:
    """Commit version, chunks, replacement and outbox together, then invalidate answers."""
    document_id = command.document_id
    versioning = VersioningResult(None, None, None, None, None)
    storage_deletes: list[str] = []
    async with uow_factory.create(master=True) as uow:
        if await lock_document_for_persist(uow, command) is None:
            log.info("Document %d deleted before outbox enqueue — aborting", document_id)
            return None

        if command.versioning is not None:
            if act_versioning_service is None:
                msg = "Versioning plan requires an act versioning service"
                raise RuntimeError(msg)

            versioning = await act_versioning_service.apply_versioning_plan(
                uow, command.versioning.profile, document_id, command.versioning.plan
            )

        await uow.documents.set_domain(document_id, command.doc_domain)

        enrich_chunks_metadata(
            command.raw_chunks,
            document_id,
            command.visibility,
            command.owner_id,
            command.group_id,
            doc_domain=command.doc_domain,
            domain_metadata=versioning.domain_metadata,
            act_version_id=versioning.act_version_id,
            act_id=versioning.act_id,
            effective_from=versioning.effective_from,
            effective_to=versioning.effective_to,
            is_current=versioning.is_current,
        )

        await process_chunks(
            uow_factory=uow_factory,
            document_id=document_id,
            filename=command.original_filename,
            chunks=command.raw_chunks,
            visibility=command.visibility,
            owner_id=command.owner_id,
            group_id=command.group_id,
            doc_domain=command.doc_domain,
            set_indexing=True,
            _existing_uow=uow,
        )

        if command.replace_id is not None:
            await apply_async_replacement(
                uow, domain_registry, command.replace_id, command.doc_domain, storage_deletes
            )

        if command.warning_message:
            await uow.documents.update_status(
                document_id,
                status=DocumentStatus.INDEXING.value,
                warning=command.warning_message,
                quality_score=command.quality.bad_ratio if command.quality else None,
            )

    # Publish cleanup keys only after commit: a rollback must preserve old files.
    command.storage_deletes.extend(storage_deletes)

    if act_versioning_service is not None:
        if versioning.act_id is not None:
            await act_versioning_service.invalidate_act_answers(versioning.act_id)
        else:
            await act_versioning_service.invalidate_document_answers(document_id)

    return versioning
