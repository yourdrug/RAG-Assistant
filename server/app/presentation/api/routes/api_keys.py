"""API Key endpoints — thin wrappers around AuthService."""

from __future__ import annotations

from application.services.auth_service import AuthService
from domain.value_objects.roles import UserKind, UserRole
from fastapi import APIRouter, Depends, HTTPException
from infrastructure.logging.actions import log_action

from presentation.api.auth_dependencies import get_current_user, require_admin
from presentation.api.dependencies import create_auth_service
from presentation.api.schemas import ApiKeyCreateRequest, ApiKeyCreateResponse, ApiKeyResponse

router = APIRouter(prefix="/clients", tags=["api-keys"])


@router.post("/{client_user_id}/api-keys", response_model=ApiKeyCreateResponse)
async def issue_api_key(
    client_user_id: int,
    req: ApiKeyCreateRequest,
    admin: dict = Depends(require_admin),
    auth_service: AuthService = Depends(create_auth_service),
):
    result = await auth_service.issue_api_key(client_user_id, name=req.name)
    log_action(
        "api_key.create", user_id=admin["id"], details={"client_user_id": client_user_id, "name": req.name}
    )
    return result


@router.get("/{client_user_id}/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(
    client_user_id: int,
    current_user: dict = Depends(get_current_user),
    auth_service: AuthService = Depends(create_auth_service),
):
    if current_user["role"] == UserRole.ADMIN or (
        current_user["kind"] == UserKind.CLIENT and current_user["id"] == client_user_id
    ):
        return await auth_service.list_api_keys(client_user_id)
    raise HTTPException(status_code=403, detail="Forbidden")


@router.delete("/{client_user_id}/api-keys/{api_key_id}")
async def revoke_api_key(
    client_user_id: int,
    api_key_id: int,
    current_user: dict = Depends(get_current_user),
    auth_service: AuthService = Depends(create_auth_service),
):
    if current_user["role"] == UserRole.ADMIN or (
        current_user["kind"] == UserKind.CLIENT and current_user["id"] == client_user_id
    ):
        await auth_service.revoke_api_key(api_key_id, client_user_id=client_user_id)
        log_action("api_key.revoke", user_id=current_user["id"], details={"api_key_id": api_key_id})
        return {"status": "revoked"}
    raise HTTPException(status_code=403, detail="Forbidden")
