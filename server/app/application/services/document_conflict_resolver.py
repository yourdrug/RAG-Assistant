"""Document conflict resolver -- application-layer orchestration for version conflicts.

Resolves conflicts between old and new document versions by delegating to
the appropriate strategy (versioning or replacement).  This is orchestration
code that opens transactions and coordinates domain services.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.services.document_versioning import decide_resolution_strategy

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
) -> str | None:
    """Resolve version conflict between new and old document.

    Called from two places:
    - upload(): if doc_domain is already known (sync path)
    - process(): after classify() determines the domain (async path)

    Returns the old document's storage key if it was deleted (caller must
    remove it from object storage after the transaction commits), or None.
    """
    is_versioned = domain_profile is not None and domain_profile.is_versioned
    strategy = decide_resolution_strategy(is_versioned)

    if strategy == "version":
        await _resolve_as_version(new_doc, old_doc, domain_profile, act_versioning_service)
        return None
    else:
        return await _resolve_as_replace(uow_factory, new_doc, old_doc)


async def _resolve_as_version(
    new_doc: Document,
    old_doc: Document,
    domain_profile: DomainProfile,
    act_versioning_service,
) -> None:
    """Versioned domain: create ActVersion, old document stays as historical."""
    log.info(
        "Versioning: doc %d is new version of doc %d (domain=%s)",
        new_doc.id,
        old_doc.id,
        domain_profile.key,
    )
    if act_versioning_service is not None:
        try:
            await act_versioning_service.process_document_versioning(
                domain_profile,
                new_doc.id,
                "",  # full_text filled by processor
            )
        except Exception:
            log.exception("Versioning failed for doc %d", new_doc.id)


async def _resolve_as_replace(
    uow_factory: UnitOfWorkFactory,
    new_doc: Document,
    old_doc: Document,
) -> str | None:
    """Legacy domain: delete old document, new takes canonical filename.

    Returns the old document's source_path for deferred S3 cleanup.
    """
    log.info(
        "Replacing: doc %d replaces doc %d (legacy domain)",
        new_doc.id,
        old_doc.id,
    )
    old_source_path = old_doc.source_path or None
    async with uow_factory.create(master=True) as uow:
        assert old_doc.id is not None
        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.DELETE_BY_DOCUMENT,
                aggregate_type="document",
                aggregate_id=old_doc.id,
                payload={"document_id": old_doc.id},
            )
        )
        await uow.documents.delete(old_doc.id)
    return old_source_path
