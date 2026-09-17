"""Group endpoints — thin wrappers around GroupService."""

from __future__ import annotations

from application.ports.rate_limit import RateLimitPolicyName
from application.services.group_service import GroupService
from fastapi import APIRouter, Depends

from presentation.api.auth_dependencies import get_current_user, require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.dependencies import create_action_logger, create_group_service
from presentation.api.schemas import (
    CreateGroupRequest,
    CurrentUser,
    GroupAssignResponse,
    GroupMemberRequest,
    GroupMemberResponse,
    GroupResponse,
)

router = APIRouter(prefix="/groups", tags=["groups"])


@router.post(
    "",
    response_model=GroupResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def create_group_endpoint(
    req: CreateGroupRequest,
    admin: CurrentUser = Depends(require_admin),
    service: GroupService = Depends(create_group_service),
    log=Depends(create_action_logger),
):
    group_id = await service.create(req.name)
    log("group.create", user_id=admin.id, details={"name": req.name})
    return GroupResponse(id=group_id, name=req.name)


@router.get("", response_model=list[GroupResponse])
async def list_groups_endpoint(
    current_user: CurrentUser = Depends(get_current_user),
    service: GroupService = Depends(create_group_service),
):
    rows = await service.list_for_user(current_user.id, current_user.role, current_user.kind)
    return [GroupResponse(id=r.id, name=r.name) for r in rows]


@router.get("/{group_id}/members", response_model=list[GroupMemberResponse])
async def get_group_members(
    group_id: int,
    admin: CurrentUser = Depends(require_admin),
    service: GroupService = Depends(create_group_service),
):
    rows = await service.list_members(group_id)
    return [GroupMemberResponse(id=r.id, email=r.email) for r in rows]


@router.post(
    "/{group_id}/members",
    response_model=GroupAssignResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def add_group_member(
    group_id: int,
    req: GroupMemberRequest,
    admin: CurrentUser = Depends(require_admin),
    service: GroupService = Depends(create_group_service),
    log=Depends(create_action_logger),
):
    await service.add_member(group_id, req.user_id)
    log("group.add_member", user_id=admin.id, details={"group_id": group_id, "user_id": req.user_id})
    return GroupAssignResponse(group_id=group_id, user_id=req.user_id)


@router.delete(
    "/{group_id}/members/{user_id}",
    response_model=GroupAssignResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def remove_group_member(
    group_id: int,
    user_id: int,
    admin: CurrentUser = Depends(require_admin),
    service: GroupService = Depends(create_group_service),
    log=Depends(create_action_logger),
):
    await service.remove_member(group_id, user_id)
    log("group.remove_member", user_id=admin.id, details={"group_id": group_id, "user_id": user_id})
    return GroupAssignResponse(group_id=group_id, user_id=user_id)
