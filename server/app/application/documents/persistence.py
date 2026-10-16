"""Document persistence -- the transactional tail of the processing pipeline.

Single transaction: set_domain → enrich chunks → bulk insert + outbox enqueue →
replacement handling → warning/status update.  Extracted from DocumentProcessor
so the transaction boundary lives in one auditable place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Unpack

from application.act_versioning import VersioningResult
from application.documents.conflicts import apply_async_replacement
from application.indexing import enrich_chunks_metadata, process_chunks
from domain.value_objects.document_status import DocumentStatus


if TYPE_CHECKING:
    from application.documents.dto import PersistDocumentResultOptions
    from application.uow import UnitOfWork, UnitOfWorkFactory

log = logging.getLogger("default")


async def persist_document_result(
    uow_factory: UnitOfWorkFactory, **options: Unpack[PersistDocumentResultOptions]
) -> VersioningResult | None:
    """Persist chunks inside a single transaction.

    Version creation, chunk persistence, replacement, outbox enqueue and status
    updates commit together. Returns the final version result, or None if the
    document was deleted while it was being processed.
    """
    document_id = options["document_id"]
    raw_chunks = options["raw_chunks"]
    visibility = options["visibility"]
    owner_id = options["owner_id"]
    group_id = options["group_id"]
    doc_domain = options["doc_domain"]
    replace_id = options["replace_id"]
    warning_message = options["warning_message"]
    quality = options["quality"]
    act_versioning_service = options.get("act_versioning_service")
    async with uow_factory.create(master=True) as uow:
        existing = await uow.documents.get_by_id(document_id)

        if existing is None:
            log.info("Document %d deleted before outbox enqueue — aborting", document_id)
            return None

        versioning = await resolve_persisted_version(uow, options)

        await uow.documents.set_domain(document_id, doc_domain)

        enrich_chunks_metadata(
            raw_chunks,
            document_id,
            visibility,
            owner_id,
            group_id,
            doc_domain=doc_domain,
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
            filename=options["original_filename"],
            chunks=raw_chunks,
            visibility=visibility,
            owner_id=owner_id,
            group_id=group_id,
            doc_domain=doc_domain,
            set_indexing=True,
            _existing_uow=uow,
        )

        if replace_id is not None:
            await apply_async_replacement(
                uow, options["domain_registry"], replace_id, doc_domain, options["storage_deletes"]
            )

        if warning_message:
            await uow.documents.update_status(
                document_id,
                DocumentStatus.INDEXING.value,
                warning=warning_message,
                quality_score=quality.bad_ratio if quality else None,
            )

    if act_versioning_service is not None:
        if versioning.act_id is not None:
            await act_versioning_service.invalidate_act_answers(versioning.act_id)
        else:
            await act_versioning_service.invalidate_document_answers(document_id)

    return versioning


async def resolve_persisted_version(
    uow: UnitOfWork, options: PersistDocumentResultOptions
) -> VersioningResult:
    document_id = options["document_id"]
    versioning = options["versioning"]
    versioning_plan = options.get("versioning_plan")
    versioning_profile = options.get("versioning_profile")
    act_versioning_service = options.get("act_versioning_service")

    if versioning_plan is not None:
        if versioning_profile is None or act_versioning_service is None:
            msg = "Versioning plan requires its profile and service"
            raise RuntimeError(msg)

        if versioning_plan.warning:
            versioning = VersioningResult(
                versioning_plan.domain_metadata,
                None,
                None,
                versioning_plan.effective_from,
                versioning_plan.warning,
            )
        else:
            act_version = await act_versioning_service.create_version_in_uow(
                uow,
                versioning_profile,
                document_id,
                versioning_plan.extracted_refs,
                effective_date=versioning_plan.effective_from,
                date_confidence=versioning_plan.date_confidence,
            )
            versioning = VersioningResult(
                versioning_plan.domain_metadata,
                act_version.id,
                act_version.act_id,
                act_version.effective_from,
                None,
                effective_to=act_version.effective_to,
                is_current=act_version.is_current,
            )

    return versioning
