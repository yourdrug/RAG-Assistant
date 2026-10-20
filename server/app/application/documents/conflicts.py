"""Select replacement targets and apply replacements inside the persistence transaction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.documents.delete import delete_document_in_uow
from domain.exceptions import EntityNotFound
from domain.value_objects.document_status import DocumentStatus


if TYPE_CHECKING:
    from application.uow import UnitOfWork
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.entities.document import Document

log = logging.getLogger(__name__)


async def resolve_upload_version_group(
    uow: UnitOfWork,
    existing: Document | None,
    replaces_document_id: int | None,
    pending_replace_id: int | None,
) -> tuple[int | None, int | None, Document | None]:
    # Keep an explicit or implicit replacement attached to its original anchor.

    if replaces_document_id is not None:
        replaces_doc = await uow.documents.get_by_id(replaces_document_id)

        if replaces_doc is None:
            msg = "Document"
            raise EntityNotFound(msg, replaces_document_id)

        version_group_id = replaces_doc.version_group_id or replaces_doc.id
        return version_group_id, replaces_doc.id, replaces_doc

    if existing and existing.status in (DocumentStatus.DONE, DocumentStatus.FAILED):
        version_group_id = existing.version_group_id or existing.id
        return version_group_id, existing.id, existing

    return None, pending_replace_id, existing


async def apply_async_replacement(
    uow: UnitOfWork,
    domain_registry: DomainProfileRegistry | None,
    replace_id: int,
    doc_domain: str,
    storage_deletes: list[str],
) -> None:
    """Delete the replaced document -- unless the domain is versioned.

    Versioned domains retain the previous edition as history.

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
        return

    await delete_document_in_uow(uow, replace_id)

    if old.source_path and old.ingestion_key is None:
        storage_deletes.append(old.source_path)
