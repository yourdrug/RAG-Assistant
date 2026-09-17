"""Admin-only curator assignment endpoints."""

from __future__ import annotations

from application.ports.rate_limit import RateLimitPolicyName
from application.services.assignment_service import AssignmentService
from fastapi import APIRouter, Depends

from presentation.api.auth_dependencies import require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.dependencies import create_action_logger, create_assignment_service
from presentation.api.schemas import CuratorScopeResponse, CurrentUser

router = APIRouter(prefix="/admin/curators", tags=["admin", "curators"])


@router.post(
    "/{curator_id}/users/{target_user_id}",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def assign_user_to_curator(
    curator_id: int,
    target_user_id: int,
    admin: CurrentUser = Depends(require_admin),
    assignment_service: AssignmentService = Depends(create_assignment_service),
    log=Depends(create_action_logger),
):
    await assignment_service.assign_user(curator_id, target_user_id, admin.id)
    log(
        "curator.assign_user",
        user_id=admin.id,
        details={"curator_id": curator_id, "target_user_id": target_user_id},
    )
    return {"status": "assigned"}


@router.delete(
    "/{curator_id}/users/{target_user_id}",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def unassign_user_from_curator(
    curator_id: int,
    target_user_id: int,
    admin: CurrentUser = Depends(require_admin),
    assignment_service: AssignmentService = Depends(create_assignment_service),
    log=Depends(create_action_logger),
):
    await assignment_service.unassign_user(curator_id, target_user_id)
    log(
        "curator.unassign_user",
        user_id=admin.id,
        details={"curator_id": curator_id, "target_user_id": target_user_id},
    )
    return {"status": "unassigned"}


@router.post(
    "/{curator_id}/groups/{group_id}",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def assign_group_to_curator(
    curator_id: int,
    group_id: int,
    admin: CurrentUser = Depends(require_admin),
    assignment_service: AssignmentService = Depends(create_assignment_service),
    log=Depends(create_action_logger),
):
    await assignment_service.assign_group(curator_id, group_id, admin.id)
    log(
        "curator.assign_group",
        user_id=admin.id,
        details={"curator_id": curator_id, "group_id": group_id},
    )
    return {"status": "assigned"}


@router.delete(
    "/{curator_id}/groups/{group_id}",
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def unassign_group_from_curator(
    curator_id: int,
    group_id: int,
    admin: CurrentUser = Depends(require_admin),
    assignment_service: AssignmentService = Depends(create_assignment_service),
    log=Depends(create_action_logger),
):
    await assignment_service.unassign_group(curator_id, group_id)
    log(
        "curator.unassign_group",
        user_id=admin.id,
        details={"curator_id": curator_id, "group_id": group_id},
    )
    return {"status": "unassigned"}


@router.get("/{curator_id}/scope", response_model=CuratorScopeResponse)
async def get_curator_scope(
    curator_id: int,
    admin: CurrentUser = Depends(require_admin),
    assignment_service: AssignmentService = Depends(create_assignment_service),
):
    scope = await assignment_service.get_scope(curator_id)
    return CuratorScopeResponse(**scope)
