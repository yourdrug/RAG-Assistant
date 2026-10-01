"""Prepare uploaded-document versioning before the persistence transaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.dto.document_processing_context import ProcessingContext, empty_versioning_result
from application.dto.versioning_dto import VersioningResult

if TYPE_CHECKING:
    from application.services.act_versioning_service import ActVersioningService
    from domain.domain_profile.registry import DomainProfileRegistry


async def prepare_processing_versioning(
    ctx: ProcessingContext,
    full_text: str,
    *,
    domain_registry: DomainProfileRegistry | None,
    act_versioning_service: ActVersioningService | None,
) -> None:
    """Prepare a plan for atomic persistence, or use the legacy service contract.

    Version creation for prepared plans stays inside persist_document_result's
    transaction together with chunk inserts, outbox enqueue and status writes.
    """
    result = empty_versioning_result()
    if domain_registry is not None and act_versioning_service is not None:
        try:
            profile = domain_registry.get(ctx.doc_domain or "")
        except KeyError:
            profile = None
        if profile is not None:
            prepare = getattr(act_versioning_service, "prepare_document_versioning", None)
            if prepare is not None:
                ctx.versioning_plan = await prepare(profile, full_text)
                if ctx.versioning_plan is not None:
                    ctx.versioning_profile = profile
                    result = VersioningResult(
                        domain_metadata=ctx.versioning_plan.domain_metadata,
                        act_version_id=None,
                        act_id=None,
                        effective_from=ctx.versioning_plan.effective_from,
                        warning=ctx.versioning_plan.warning,
                    )
            else:
                result = await act_versioning_service.process_document_versioning(
                    profile, ctx.document_id, full_text
                )
    ctx.versioning = result
    if result.warning:
        ctx.warnings.append(result.warning)
