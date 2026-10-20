"""Prepare uploaded-document versioning before the persistence transaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.documents.dto import PreparedDocumentVersioning


if TYPE_CHECKING:
    from application.act_versioning import ActVersioningService
    from domain.ingestion.profiles.registry import DomainProfileRegistry


async def prepare_processing_versioning(
    doc_domain: str,
    full_text: str,
    *,
    domain_registry: DomainProfileRegistry | None,
    act_versioning_service: ActVersioningService | None,
) -> PreparedDocumentVersioning | None:
    if domain_registry is None or act_versioning_service is None:
        return None

    try:
        profile = domain_registry.get(doc_domain)
    except KeyError:
        return None

    plan = await act_versioning_service.prepare_document_versioning(profile, full_text)
    return PreparedDocumentVersioning(plan, profile) if plan is not None else None
