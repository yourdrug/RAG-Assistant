"""
api/routes/clients.py — Client assignment endpoints with Pydantic response models.
"""

from fastapi import APIRouter, Depends, HTTPException
from infrastructure.auth import require_admin
from infrastructure.database import (
    assign_client,
    get_db,
    get_user_by_id,
    list_assignments_for_client,
    unassign_client,
)
from sqlalchemy.orm import Session

from api.schemas import AssignClientRequest

router = APIRouter(prefix="/clients", tags=["clients"])


@router.post("/{client_user_id}/assignments")
async def assign_client_endpoint(
    client_user_id: int,
    req: AssignClientRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    client_user = get_user_by_id(db, client_user_id)
    internal_user = get_user_by_id(db, req.internal_user_id)
    if client_user is None or client_user["kind"] != "client":
        raise HTTPException(status_code=400, detail="client_user_id must be a user with kind='client'")
    if internal_user is None or internal_user["kind"] != "internal":
        raise HTTPException(status_code=400, detail="internal_user_id must be a user with kind='internal'")

    assign_client(db, req.internal_user_id, client_user_id, admin["id"])
    return {"client_user_id": client_user_id, "internal_user_id": req.internal_user_id}


@router.delete("/{client_user_id}/assignments/{internal_user_id}")
async def unassign_client_endpoint(
    client_user_id: int,
    internal_user_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    unassign_client(db, internal_user_id, client_user_id)
    return {"status": "removed"}


@router.get("/{client_user_id}/assignments")
async def list_client_assignments_endpoint(
    client_user_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return list_assignments_for_client(db, client_user_id)
