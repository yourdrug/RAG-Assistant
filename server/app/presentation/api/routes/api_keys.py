"""API Key endpoints — thin wrappers around AuthService."""

from __future__ import annotations

from application.services.auth_service import AuthService
from fastapi import APIRouter, Depends, HTTPException

from presentation.api.auth_dependencies import get_current_user, require_admin
from presentation.api.dependencies import create_auth_service
from presentation.api.schemas import ApiKeyCreatedResponse, ApiKeyCreateRequest, ApiKeyResponse

router = APIRouter(prefix="/clients", tags=["api-keys"])


@router.post("/{client_user_id}/api-keys", response_model=ApiKeyCreatedResponse)
async def issue_api_key(
    client_user_id: int,
    req: ApiKeyCreateRequest,
    admin: dict = Depends(require_admin),
    auth_service: AuthService = Depends(create_auth_service),
):
    return await auth_service.issue_api_key(client_user_id, name=req.name)


@router.get("/{client_user_id}/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(
    client_user_id: int,
    current_user: dict = Depends(get_current_user),
    auth_service: AuthService = Depends(create_auth_service),
):
    if current_user["role"] == "admin" or (
        current_user["kind"] == "client" and current_user["id"] == client_user_id
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
    if current_user["role"] == "admin" or (
        current_user["kind"] == "client" and current_user["id"] == client_user_id
    ):
        return await auth_service.revoke_api_key(api_key_id, client_user_id=client_user_id)
    raise HTTPException(status_code=403, detail="Forbidden")
