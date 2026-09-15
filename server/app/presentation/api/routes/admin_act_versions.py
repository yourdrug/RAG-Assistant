"""Admin act version management — review and edit regulatory act versions."""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from application.services.act_versioning_service import ActVersioningService

from presentation.api.auth_dependencies import require_admin
from presentation.api.dependencies import create_act_versioning_service

logger = logging.getLogger("default")

router = APIRouter(tags=["admin-act-versions"])


class ActVersionReviewItem(BaseModel):
    id: int
    act_id: int | None
    document_id: int
    document_filename: str
    effective_from: date | None
    effective_to: date | None
    is_current: bool
    date_source: str
    date_confidence: float | None


class ActSummary(BaseModel):
    id: int
    act_type: str
    act_number: str | None
    title: str


class ActVersionListResponse(BaseModel):
    versions: list[ActVersionReviewItem]
    total: int
    acts: list[ActSummary] = []


class ActVersionUpdateRequest(BaseModel):
    effective_from: date | None = None
    effective_to: date | None = None
    act_id: int | None = None


@router.get("/admin/act-versions", response_model=ActVersionListResponse)
async def list_act_versions(
    review: str | None = None,
    admin: dict = Depends(require_admin),
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
    admin: dict = Depends(require_admin),
    service: ActVersioningService = Depends(create_act_versioning_service),
):
    """Update act version dates/linkage. Sets date_source='manual'."""
    await service.update_version(
        version_id=version_id,
        effective_from=body.effective_from,
        effective_to=body.effective_to,
        act_id=body.act_id,
        verified_by=admin["id"],
    )
    return {"status": "updated", "version_id": version_id}
