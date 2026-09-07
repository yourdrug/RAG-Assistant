"""Document versioning — resolves conflicts between old and new document versions.

Separates physical conflict resolution (rename, always immediate) from
logical policy (version vs replace, requires domain knowledge).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from application.ports.unit_of_work_factory import UnitOfWorkFactory
    from domain.domain_profile.protocol import DomainProfile
    from domain.entities.document import Document

log = logging.getLogger(__name__)


async def resolve_conflict(
    uow_factory: UnitOfWorkFactory,
    new_doc: Document,
    old_doc: Document,
    domain_profile: DomainProfile | None,
    *,
    act_versioning_service=None,
) -> None:
    """Resolve version conflict between new and old document.

    Called from two places:
    - upload(): if doc_domain is already known (sync path)
    - process(): after classify() determines the domain (async path)

    For versioned domains: creates ActVersion, old document stays.
    For legacy domains: deletes old, new takes canonical filename.
    """
    if domain_profile is not None and domain_profile.is_versioned:
        await _resolve_as_version(
            uow_factory, new_doc, old_doc, domain_profile, act_versioning_service,
        )
    else:
        await _resolve_as_replace(uow_factory, new_doc, old_doc)


async def _resolve_as_version(
    uow_factory: UnitOfWorkFactory,
    new_doc: Document,
    old_doc: Document,
    domain_profile: DomainProfile,
    act_versioning_service,
) -> None:
    """Versioned domain: create ActVersion, old document stays as historical."""
    log.info(
        "Versioning: doc %d is new version of doc %d (domain=%s)",
        new_doc.id, old_doc.id, domain_profile.key,
    )
    if act_versioning_service is not None:
        try:
            await act_versioning_service.process_document_versioning(
                domain_profile, new_doc.id, "",  # full_text filled by processor
            )
        except Exception:
            log.exception("Versioning failed for doc %d", new_doc.id)


async def _resolve_as_replace(
    uow_factory: UnitOfWorkFactory,
    new_doc: Document,
    old_doc: Document,
) -> None:
    """Legacy domain: delete old document, new takes canonical filename."""
    log.info(
        "Replacing: doc %d replaces doc %d (legacy domain)",
        new_doc.id, old_doc.id,
    )
    async with uow_factory.create(master=True) as uow:
        # Delete old document's chunks via outbox
        from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry

        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.DELETE_BY_DOCUMENT,
                aggregate_type="document",
                aggregate_id=old_doc.id,
                payload={"document_id": old_doc.id},
            )
        )
        await uow.documents.delete(old_doc.id)
