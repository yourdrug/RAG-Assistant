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

from api.schemas import CreateGroupRequest, GroupMemberRequest

router = APIRouter(prefix="/groups", tags=["groups"])


@router.post("")
async def create_group_endpoint(
    req: CreateGroupRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    group_id = create_group(db, req.name)
    return {"id": group_id, "name": req.name}


@router.get("")
async def list_groups_endpoint(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user["role"] == "admin":
        return list_groups(db)
    if current_user["kind"] != "internal":
        return []
    return list_groups(db, only_ids=get_user_group_ids(db, current_user["id"]))


@router.get("/{group_id}/members")
async def get_group_members(
    group_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return list_group_members(db, group_id)


@router.post("/{group_id}/members")
async def add_group_member(
    group_id: int,
    req: GroupMemberRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    target = get_user_by_id(db, req.user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    if target["kind"] != "internal":
        raise HTTPException(status_code=400, detail="В группы добавляются только internal-сотрудники")
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
