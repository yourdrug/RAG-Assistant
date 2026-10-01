"""Document conflict resolver -- application-layer orchestration for version conflicts.

Resolves conflicts between old and new document versions by delegating to
the appropriate strategy (versioning or replacement).  This is orchestration
code that opens transactions and coordinates domain services.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.services.document_pipeline import enqueue_delete_by_document
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
) -> str | None:
    """Resolve version conflict between new and old document.

    Called from two places:
    - upload(): if doc_domain is already known (sync path)
    - process(): after classify() determines the domain (async path)

    Returns the old document's storage key if it was deleted (caller must
    remove it from object storage after the transaction commits), or None.

    Versioning is NOT performed here: full text isn't available at conflict
    time, and DocumentProcessor runs process_document_versioning exactly once
    (with the real text) right after classification.
    """
    is_versioned = domain_profile is not None and domain_profile.is_versioned
    strategy = decide_resolution_strategy(is_versioned)

    if strategy == "version":
        if domain_profile is None:
            raise RuntimeError("domain_profile is None for versioned resolution")
        _resolve_as_version(new_doc, old_doc, domain_profile)
        return None
    else:
        async with uow_factory.create(master=True) as uow:
            return await resolve_conflict_in_uow(uow, new_doc, old_doc, domain_profile)


async def resolve_conflict_in_uow(
    uow,
    new_doc: Document,
    old_doc: Document,
    domain_profile: DomainProfile | None,
) -> str | None:
    """Resolve a conflict inside the caller's transaction."""
    is_versioned = domain_profile is not None and domain_profile.is_versioned
    if is_versioned:
        if domain_profile is None:
            raise RuntimeError("domain_profile is None for versioned resolution")
        _resolve_as_version(new_doc, old_doc, domain_profile)
        return None
    return await _resolve_as_replace(uow, new_doc, old_doc)


def _resolve_as_version(
    new_doc: Document,
    old_doc: Document,
    domain_profile: DomainProfile,
) -> None:
    """Versioned domain: the old document stays as historical.

    The new document's ActVersion is created later by DocumentProcessor
    (with the parsed full text) -- see ActVersioningService.
    """
    log.info(
        "Versioning: doc %d is new version of doc %d (domain=%s)",
        new_doc.id,
        old_doc.id,
        domain_profile.key,
    )


async def _resolve_as_replace(
    uow,
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
    if old_doc.id is None:
        raise RuntimeError("old_doc id is None")
    await enqueue_delete_by_document(uow, old_doc.id)
    await uow.documents.delete(old_doc.id)
    return old_source_path


async def apply_async_replacement(
    uow,
    domain_registry,
    replace_id: int,
    doc_domain: str,
    storage_deletes: list[str],
) -> None:
    """Delete the replaced document -- unless the domain is versioned.

    Versioned domain: the previous edition must stay -- its ActVersion was
    already flipped to is_current=False by handle_versioned_upload for the
    new document.

    Called from the async persist path (persist_document_result) inside the
    same transaction that indexes the new document; the S3 object listed in
    ``storage_deletes`` is removed by the caller AFTER the commit.
    """
    replace_profile = domain_registry.get(doc_domain) if domain_registry is not None else None
    if replace_profile is not None and replace_profile.is_versioned:
        log.info("Keeping previous edition doc %d (versioned domain %s)", replace_id, doc_domain)
        return
    old = await uow.documents.get_by_id(replace_id)
    if old is None:
        # Already removed by the sync conflict path (resolve_conflict) before
        # this transaction -- enqueueing again would send a duplicate
        # DELETE_BY_DOCUMENT to the outbox.
        log.info("Replacement target %d already deleted — skipping duplicate outbox entry", replace_id)
        return
    await enqueue_delete_by_document(uow, replace_id)
    if old.source_path:
        # the S3 object is deleted after the transaction commits.
        storage_deletes.append(old.source_path)
    await uow.documents.delete(replace_id)
