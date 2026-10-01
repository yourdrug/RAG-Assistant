"""Document persistence -- the transactional tail of the processing pipeline.

Single transaction: set_domain → enrich chunks → bulk insert + outbox enqueue →
replacement handling → warning/status update.  Extracted from DocumentProcessor
so the transaction boundary lives in one auditable place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.dto.versioning_dto import VersioningPlan, VersioningResult
from application.services.document_conflict_resolver import apply_async_replacement
from application.services.document_pipeline import enrich_chunks_metadata, process_chunks
from domain.value_objects.document_status import DocumentStatus

if TYPE_CHECKING:
    from application.ports.unit_of_work_factory import UnitOfWorkFactory
    from application.services.act_versioning_service import ActVersioningService
    from domain.domain_profile.protocol import DomainProfile
    from domain.value_objects.pdf_quality_report import PDFQualityReport

log = logging.getLogger("default")


async def persist_document_result(
    uow_factory: UnitOfWorkFactory,
    *,
    document_id: int,
    original_filename: str,
    raw_chunks: list,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    doc_domain: str,
    replace_id: int | None,
    warning_message: str | None,
    quality: PDFQualityReport | None,
    storage_deletes: list[str],
    domain_registry,
    versioning: VersioningResult,
    versioning_plan: VersioningPlan | None = None,
    versioning_profile: "DomainProfile | None" = None,
    act_versioning_service: "ActVersioningService | None" = None,
) -> VersioningResult | None:
    """Persist chunks inside a single transaction.

    Version creation, chunk persistence, replacement, outbox enqueue and status
    updates commit together. Returns the final version result, or None if the
    document was deleted while it was being processed.
    """
    async with uow_factory.create(master=True) as uow:
        existing = await uow.documents.get_by_id(document_id)
        if existing is None:
            log.info("Document %d deleted before outbox enqueue — aborting", document_id)
            return None

        if versioning_plan is not None:
            if versioning_profile is None or act_versioning_service is None:
                raise RuntimeError("Versioning plan requires its profile and service")
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
                    versioning_plan.effective_date,
                    versioning_plan.date_confidence,
                )
                versioning = VersioningResult(
                    versioning_plan.domain_metadata,
                    act_version.id,
                    act_version.act_id,
                    versioning_plan.effective_from,
                    None,
                )

        await uow.documents.set_domain(document_id, doc_domain)

        enrich_chunks_metadata(
            raw_chunks,
            document_id,
            visibility,
            owner_id,
            group_id,
            doc_domain,
            domain_metadata=versioning.domain_metadata,
            act_version_id=versioning.act_version_id,
            act_id=versioning.act_id,
            effective_from=versioning.effective_from,
        )

        await process_chunks(
            uow_factory=uow_factory,
            document_id=document_id,
            filename=original_filename,
            chunks=raw_chunks,
            visibility=visibility,
            owner_id=owner_id,
            group_id=group_id,
            doc_domain=doc_domain,
            set_indexing=True,
            _existing_uow=uow,
        )

        if replace_id is not None:
            await apply_async_replacement(uow, domain_registry, replace_id, doc_domain, storage_deletes)

        if warning_message:
            await uow.documents.update_status(
                document_id,
                DocumentStatus.INDEXING.value,
                warning=warning_message,
                quality_score=quality.bad_ratio if quality else None,
            )
    return versioning
