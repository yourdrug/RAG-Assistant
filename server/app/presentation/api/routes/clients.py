"""Client assignment endpoints — thin wrappers around ClientAssignmentRepository."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from infrastructure.auth.fastapi_dependencies import require_admin
from infrastructure.database.session import get_db
from sqlalchemy.orm import Session

from presentation.api.dependencies import get_repos
from presentation.api.schemas import AssignClientRequest

router = APIRouter(prefix="/clients", tags=["clients"])


@router.post("/{client_user_id}/assignments")
async def assign_client_endpoint(
    client_user_id: int,
    req: AssignClientRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    client_user = repos["user_repo"].get_by_id(client_user_id)
    internal_user = repos["user_repo"].get_by_id(req.internal_user_id)
    if client_user is None or client_user.kind != "client":
        raise HTTPException(status_code=400, detail="client_user_id must be a user with kind='client'")
    if internal_user is None or internal_user.kind != "internal":
        raise HTTPException(status_code=400, detail="internal_user_id must be a user with kind='internal'")

    repos["client_assignment_repo"].assign(req.internal_user_id, client_user_id, admin["id"])
    return {"client_user_id": client_user_id, "internal_user_id": req.internal_user_id}


@router.delete("/{client_user_id}/assignments/{internal_user_id}")
async def unassign_client_endpoint(
    client_user_id: int,
    internal_user_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    repos["client_assignment_repo"].unassign(internal_user_id, client_user_id)
    return {"status": "removed"}


@router.get("/{client_user_id}/assignments")
async def list_client_assignments_endpoint(
    client_user_id: int,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    repos = get_repos(db)
    return repos["client_assignment_repo"].list_for_client(client_user_id)
