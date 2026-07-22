"""Group endpoints — thin wrappers around GroupRepository."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from infrastructure.auth.fastapi_dependencies import get_current_user, require_admin
from infrastructure.database.session import get_db
from sqlalchemy.orm import Session

from presentation.api.dependencies import get_repos
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
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    group_id = repos["group_repo"].create(req.name)
    return GroupResponse(id=group_id, name=req.name)


@router.get("", response_model=list[GroupResponse])
async def list_groups_endpoint(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    if current_user["role"] == "admin":
        rows = repos["group_repo"].list_all()
    elif current_user["kind"] != "internal":
        rows = []
    else:
        group_ids = repos["group_repo"].get_user_group_ids(current_user["id"])
        rows = repos["group_repo"].list_by_ids(group_ids) if group_ids else []
    return [GroupResponse(id=r["id"], name=r["name"]) for r in rows]


@router.get("/{group_id}/members", response_model=list[GroupMemberResponse])
async def get_group_members(
    group_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    rows = repos["group_repo"].list_members(group_id)
    return [GroupMemberResponse(id=r["id"], email=r["email"]) for r in rows]


@router.post("/{group_id}/members")
async def add_group_member(
    group_id: int,
    req: GroupMemberRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    target = repos["user_repo"].get_by_id(req.user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    if target.kind != "internal":
        raise HTTPException(status_code=400, detail="Only internal employees can be added to groups")
    repos["group_repo"].add_user(req.user_id, group_id)
    return {"group_id": group_id, "user_id": req.user_id}


@router.delete("/{group_id}/members/{user_id}")
async def remove_group_member(
    group_id: int,
    user_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    repos["group_repo"].remove_user(user_id, group_id)
    return {"group_id": group_id, "user_id": user_id}
