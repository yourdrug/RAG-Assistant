"""Admin act version management — review and edit regulatory act versions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from application.services.act_versioning_service import ActVersioningService

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_act_versioning_service
from presentation.api.schemas import (
    ActVersionListResponse,
    ActVersionReviewItem,
    ActVersionUpdateRequest,
    CurrentUser,
)

logger = logging.getLogger("default")

router = APIRouter(tags=["admin-act-versions"])


@router.get("/admin/act-versions", response_model=ActVersionListResponse)
async def list_act_versions(
    review: str | None = None,
    admin: CurrentUser = Depends(require_admin),
    service: ActVersioningService = Depends(create_act_versioning_service),
):
    """List act versions pending review (date_source='extracted' or act_id IS NULL)."""
    if review != "pending":
        return ActVersionListResponse(versions=[], total=0)

    versions = await service.list_pending_review()

    doc_ids = list({v.document_id for v in versions})
    doc_filenames = await service.get_document_filenames(doc_ids) if doc_ids else {}

    acts = await service.list_recent_acts()

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
        total=len(versions),
        acts=acts,
    )


@router.patch("/admin/act-versions/{version_id}")
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
    )
    return {"status": "updated", "version_id": version_id}
