"""API Key endpoints — thin wrappers around AuthService."""

from __future__ import annotations

from application.ports.rate_limit import RateLimitPolicyName
from application.services.auth_service import AuthService
from domain.value_objects.roles import UserKind, UserRole
from fastapi import APIRouter, Depends, HTTPException

from presentation.api.auth_dependencies import get_current_user, require_admin
from presentation.api.rate_limit import rate_limit
from presentation.api.dependencies import create_action_logger, create_auth_service
from presentation.api.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyRevokeResponse,
    ApiKeyResponse,
    CurrentUser,
)

router = APIRouter(prefix="/clients", tags=["api-keys"])


def _check_client_access(current_user: CurrentUser, client_user_id: int) -> None:
    """Raise 403 if current user is not admin and not the owning client."""
    if current_user.role == UserRole.ADMIN or (
        current_user.kind == UserKind.CLIENT and current_user.id == client_user_id
    ):
        return
    raise HTTPException(status_code=403, detail="Forbidden")


@router.post(
    "/{client_user_id}/api-keys",
    response_model=ApiKeyCreateResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def issue_api_key(
    client_user_id: int,
    req: ApiKeyCreateRequest,
    admin: CurrentUser = Depends(require_admin),
    auth_service: AuthService = Depends(create_auth_service),
    log=Depends(create_action_logger),
):
    result = await auth_service.issue_api_key(client_user_id, name=req.name)
    log("api_key.create", user_id=admin.id, details={"client_user_id": client_user_id, "name": req.name})
    return ApiKeyCreateResponse(key=result.api_key, key_prefix=result.key_prefix, name=result.name)


@router.get("/{client_user_id}/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(
    client_user_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    auth_service: AuthService = Depends(create_auth_service),
):
    _check_client_access(current_user, client_user_id)
    keys = await auth_service.list_api_keys(client_user_id)
    return [
        ApiKeyResponse(
            id=k.id,
            key_prefix=k.key_prefix,
            name=k.name,
            creation_date=k.creation_date,
            last_used_at=k.last_used_at,
            revoked_at=k.revoked_at,
            is_active=k.is_active,
        )
        for k in keys
    ]


@router.delete(
    "/{client_user_id}/api-keys/{api_key_id}",
    response_model=ApiKeyRevokeResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def revoke_api_key(
    client_user_id: int,
    api_key_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    auth_service: AuthService = Depends(create_auth_service),
    log=Depends(create_action_logger),
):
    _check_client_access(current_user, client_user_id)
    await auth_service.revoke_api_key(api_key_id, client_user_id=client_user_id)
    log("api_key.revoke", user_id=current_user.id, details={"api_key_id": api_key_id})
    return ApiKeyRevokeResponse(status="revoked")
