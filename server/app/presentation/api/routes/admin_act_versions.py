"""Admin act version management — review and edit regulatory act versions."""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from application.services.act_versioning_service import ActVersioningService

from presentation.api.auth_dependencies import require_admin

logger = logging.getLogger("default")

router = APIRouter(tags=["admin-act-versions"])


class ActVersionReviewItem(BaseModel):
    id: int
    act_id: int | None
    document_id: int
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


def _get_act_versioning_service(request: Request) -> ActVersioningService:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("Container not initialized")
    return ActVersioningService(
        uow_factory=container.infrastructure.uow_factory,
        settings=container.infrastructure.domain_settings,
    )


async def _list_recent_acts(request: Request, limit: int = 20) -> list[ActSummary]:
    """Recent acts — linkage suggestions for versions pending manual review."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        return []
    async with container.infrastructure.uow_factory.create() as uow:
        acts = await uow.regulatory_acts.list_all()
    return [
        ActSummary(id=a.id, act_type=a.act_type, act_number=a.act_number, title=a.title)
        for a in acts[-limit:]
    ]


@router.get("/admin/act-versions", response_model=ActVersionListResponse)
async def list_act_versions(
    request: Request,
    review: str | None = None,
    admin: dict = Depends(require_admin),
):
    """List act versions pending review (date_source='extracted' or act_id IS NULL)."""
    service = _get_act_versioning_service(request)
    if review != "pending":
        return ActVersionListResponse(versions=[], total=0)

    versions = await service.list_pending_review()
    return ActVersionListResponse(
        versions=[
            ActVersionReviewItem(
                id=v.id,  # type: ignore[arg-type]
                act_id=v.act_id,
                document_id=v.document_id,
                effective_from=v.effective_from,
                effective_to=v.effective_to,
                is_current=v.is_current,
                date_source=v.date_source,
                date_confidence=v.date_confidence,
            )
            for v in versions
        ],
        total=len(versions),
        acts=await _list_recent_acts(request),
    )


@router.patch("/admin/act-versions/{version_id}")
async def update_act_version(
    version_id: int,
    body: ActVersionUpdateRequest,
    request: Request,
    admin: dict = Depends(require_admin),
):
    """Update act version dates/linkage. Sets date_source='manual'."""
    service = _get_act_versioning_service(request)
    await service.update_version(
        version_id=version_id,
        effective_from=body.effective_from,
        effective_to=body.effective_to,
        act_id=body.act_id,
        verified_by=admin["id"],
    )
    return {"status": "updated", "version_id": version_id}
