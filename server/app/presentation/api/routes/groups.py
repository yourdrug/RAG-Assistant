"""Group endpoints — thin wrappers around GroupRepository."""

from __future__ import annotations

from application.uow import UnitOfWork
from domain.value_objects.roles import UserKind, UserRole
from fastapi import APIRouter, Depends, HTTPException
from infrastructure.logging.actions import log_action

from presentation.api.auth_dependencies import get_current_user, require_admin
from presentation.api.dependencies import get_uow
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
    uow: UnitOfWork = Depends(get_uow),
):
    group_id = await uow.groups.create(req.name)
    log_action("group.create", user_id=admin["id"], details={"name": req.name})
    return GroupResponse(id=group_id, name=req.name)


@router.get("", response_model=list[GroupResponse])
async def list_groups_endpoint(
    current_user: dict = Depends(get_current_user),
    uow: UnitOfWork = Depends(get_uow),
):
    if current_user["role"] == UserRole.ADMIN:
        rows = await uow.groups.list_all()
    elif current_user["kind"] != UserKind.INTERNAL:
        rows = []
    else:
        group_ids = await uow.groups.get_user_group_ids(current_user["id"])
        rows = await uow.groups.list_by_ids(group_ids) if group_ids else []
    return [GroupResponse(id=r["id"], name=r["name"]) for r in rows]


@router.get("/{group_id}/members", response_model=list[GroupMemberResponse])
async def get_group_members(
    group_id: int,
    admin: dict = Depends(require_admin),
    uow: UnitOfWork = Depends(get_uow),
):
    rows = await uow.groups.list_members(group_id)
    return [GroupMemberResponse(id=r["id"], email=r["email"]) for r in rows]


@router.post("/{group_id}/members")
async def add_group_member(
    group_id: int,
    req: GroupMemberRequest,
    admin: dict = Depends(require_admin),
    uow: UnitOfWork = Depends(get_uow),
):
    target = await uow.users.get_by_id(req.user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    if target.kind != UserKind.INTERNAL:
        raise HTTPException(status_code=400, detail="Only internal employees can be added to groups")
    await uow.groups.add_user(req.user_id, group_id)
    log_action("group.add_member", user_id=admin["id"], details={"group_id": group_id, "user_id": req.user_id})
    return {"group_id": group_id, "user_id": req.user_id}


@router.delete("/{group_id}/members/{user_id}")
async def remove_group_member(
    group_id: int,
    user_id: int,
    admin: dict = Depends(require_admin),
    uow: UnitOfWork = Depends(get_uow),
):
    await uow.groups.remove_user(user_id, group_id)
    log_action("group.remove_member", user_id=admin["id"], details={"group_id": group_id, "user_id": user_id})
    return {"group_id": group_id, "user_id": user_id}
