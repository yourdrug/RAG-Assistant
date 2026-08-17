"""Group endpoints — thin wrappers around GroupService."""

from __future__ import annotations

from application.services.group_service import GroupService
from fastapi import APIRouter, Depends, HTTPException
from infrastructure.logging.actions import log_action

from presentation.api.auth_dependencies import get_current_user, require_admin
from presentation.api.dependencies import get_group_service
from presentation.api.schemas import (
    CreateGroupRequest,
    GroupMemberRequest,
    GroupMemberResponse,
    GroupResponse,
)

router = APIRouter(prefix="/groups", tags=["groups"])


@router.post("", response_model=GroupResponse)
async def create_group_endpoint(
    req: CreateGroupRequest,
    admin: dict = Depends(require_admin),
    service: GroupService = Depends(get_group_service),
):
    group_id = await service.create(req.name)
    log_action("group.create", user_id=admin["id"], details={"name": req.name})
    return GroupResponse(id=group_id, name=req.name)


@router.get("", response_model=list[GroupResponse])
async def list_groups_endpoint(
    current_user: dict = Depends(get_current_user),
    service: GroupService = Depends(get_group_service),
):
    rows = await service.list_for_user(current_user["id"], current_user["role"], current_user["kind"])
    return [GroupResponse(id=r.id, name=r.name) for r in rows]


@router.get("/{group_id}/members", response_model=list[GroupMemberResponse])
async def get_group_members(
    group_id: int,
    admin: dict = Depends(require_admin),
    service: GroupService = Depends(get_group_service),
):
    rows = await service.list_members(group_id)
    return [GroupMemberResponse(id=r.id, email=r.email) for r in rows]


@router.post("/{group_id}/members")
async def add_group_member(
    group_id: int,
    req: GroupMemberRequest,
    admin: dict = Depends(require_admin),
    service: GroupService = Depends(get_group_service),
):
    try:
        await service.add_member(group_id, req.user_id)
    except Exception as e:
        from domain.exceptions import EntityNotFound, ValidationError

        if isinstance(e, EntityNotFound):
            raise HTTPException(status_code=404, detail="User not found")
        if isinstance(e, ValidationError):
            raise HTTPException(status_code=400, detail=str(e.detail))
        raise
    log_action(
        "group.add_member", user_id=admin["id"], details={"group_id": group_id, "user_id": req.user_id}
    )
    return {"group_id": group_id, "user_id": req.user_id}


@router.delete("/{group_id}/members/{user_id}")
async def remove_group_member(
    group_id: int,
    user_id: int,
    admin: dict = Depends(require_admin),
    service: GroupService = Depends(get_group_service),
):
    await service.remove_member(group_id, user_id)
    log_action("group.remove_member", user_id=admin["id"], details={"group_id": group_id, "user_id": user_id})
    return {"group_id": group_id, "user_id": user_id}
