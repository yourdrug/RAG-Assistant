"""Document persistence -- the transactional tail of the processing pipeline.

Single transaction: set_domain → enrich chunks → bulk insert + outbox enqueue →
replacement handling → warning/status update.  Extracted from DocumentProcessor
so the transaction boundary lives in one auditable place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.services.document_conflict_resolver import apply_async_replacement
from application.services.document_pipeline import enrich_chunks_metadata, process_chunks
from domain.value_objects.document_status import DocumentStatus

if TYPE_CHECKING:
    from datetime import date

    from application.ports.unit_of_work_factory import UnitOfWorkFactory
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
    domain_metadata: dict | None,
    act_version_id: int | None,
    act_id: int | None,
    effective_from: "date | str | None",
    replace_id: int | None,
    warning_message: str | None,
    quality: PDFQualityReport | None,
    storage_deletes: list[str],
    domain_registry,
) -> bool:
    """Persist chunks inside a single transaction.

    set_domain, enrich, process_chunks, replacement, and warning/status update.

    Returns True on success, False if the document was deleted mid-flight.
    """
    async with uow_factory.create(master=True) as uow:
        existing = await uow.documents.get_by_id(document_id)
        if existing is None:
            log.info("Document %d deleted before outbox enqueue — aborting", document_id)
            return False

        await uow.documents.set_domain(document_id, doc_domain)

        enrich_chunks_metadata(
            raw_chunks,
            document_id,
            visibility,
            owner_id,
            group_id,
            doc_domain,
            domain_metadata=domain_metadata,
            act_version_id=act_version_id,
            act_id=act_id,
            effective_from=effective_from,
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

    return True
