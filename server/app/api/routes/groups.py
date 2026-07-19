"""
api/routes/groups.py — Group endpoints with Pydantic response models.
"""

from fastapi import APIRouter, Depends, HTTPException
from infrastructure.auth import get_current_user, require_admin
from infrastructure.database import (
    add_user_to_group,
    create_group,
    get_db,
    get_user_by_id,
    get_user_group_ids,
    list_group_members,
    list_groups,
    remove_user_from_group,
)
from sqlalchemy.orm import Session

from api.schemas import CreateGroupRequest, GroupMemberRequest, GroupMemberResponse, GroupResponse

router = APIRouter(prefix="/groups", tags=["groups"])


@router.post("", response_model=GroupResponse)
async def create_group_endpoint(
    req: CreateGroupRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    group_id = create_group(db, req.name)
    return GroupResponse(id=group_id, name=req.name)


@router.get("", response_model=list[GroupResponse])
async def list_groups_endpoint(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] == "admin":
        rows = list_groups(db)
    elif current_user["kind"] != "internal":
        rows = []
    else:
        rows = list_groups(db, only_ids=get_user_group_ids(db, current_user["id"]))
    return [GroupResponse(id=r["id"], name=r["name"]) for r in rows]


@router.get("/{group_id}/members", response_model=list[GroupMemberResponse])
async def get_group_members(
    group_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    rows = list_group_members(db, group_id)
    return [GroupMemberResponse(id=r["id"], email=r["email"]) for r in rows]


@router.post("/{group_id}/members")
async def add_group_member(
    group_id: int,
    req: GroupMemberRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    target = get_user_by_id(db, req.user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    if target["kind"] != "internal":
        raise HTTPException(status_code=400, detail="Only internal employees can be added to groups")
    add_user_to_group(db, req.user_id, group_id)
    return {"group_id": group_id, "user_id": req.user_id}


@router.delete("/{group_id}/members/{user_id}")
async def remove_group_member(
    group_id: int,
    user_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    remove_user_from_group(db, user_id, group_id)
    return {"group_id": group_id, "user_id": user_id}
