"""Admin act version management — review and edit regulatory act versions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from application.ports.rate_limit import RateLimitPolicyName
from application.services.act_versioning_service import ActVersioningService

from presentation.api.auth_dependencies import require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.dependencies import create_act_versioning_service
from presentation.api.schemas import (
    ActVersionListResponse,
    ActVersionReviewItem,
    ActVersionUpdateRequest,
    ActVersionUpdateResponse,
    CurrentUser,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["admin-act-versions"])


@router.get("/admin/act-versions", response_model=ActVersionListResponse)
async def list_act_versions(
    review: str | None = None,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    admin: CurrentUser = Depends(require_admin),
    service: ActVersioningService = Depends(create_act_versioning_service),
):
    """List act versions pending review (date_source='extracted' or act_id IS NULL)."""
    if review not in ("pending", "all"):
        return ActVersionListResponse(versions=[], total=0)

    if review == "all":
        versions, total = await service.list_versions_page(limit, offset)
    else:
        versions = await service.list_pending_review()
        total = len(versions)

    doc_ids = list({v.document_id for v in versions})
    doc_filenames = await service.get_document_filenames(doc_ids) if doc_ids else {}

    acts = await service.list_recent_acts(
        include_act_ids={v.act_id for v in versions if v.act_id is not None}
    )

    return ActVersionListResponse(
        versions=[
            ActVersionReviewItem(
                id=v.id,  # type: ignore[arg-type]
                act_id=v.act_id,
                document_id=v.document_id,
                document_filename=doc_filenames.get(v.document_id, f"#{v.document_id}"),
                effective_from=v.effective_from,
                effective_to=v.effective_to,
                is_current=v.is_current,
                date_source=v.date_source,
                date_confidence=v.date_confidence,
            )
            for v in versions
        ],
        total=total,
        next_offset=(offset + len(versions) if review == "all" and offset + len(versions) < total else None),
        acts=acts,
    )


@router.patch(
    "/admin/act-versions/{version_id}",
    response_model=ActVersionUpdateResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def update_act_version(
    version_id: int,
    body: ActVersionUpdateRequest,
    admin: CurrentUser = Depends(require_admin),
    service: ActVersioningService = Depends(create_act_versioning_service),
):
    """Update act version dates/linkage. Sets date_source='manual'."""
    await service.update_version(
        version_id=version_id,
        effective_from=body.effective_from,
        effective_to=body.effective_to,
        act_id=body.act_id,
        verified_by=admin.id,
        effective_from_provided="effective_from" in body.model_fields_set,
        effective_to_provided="effective_to" in body.model_fields_set,
        act_id_provided="act_id" in body.model_fields_set,
    )
    return ActVersionUpdateResponse(status="updated", version_id=version_id)
