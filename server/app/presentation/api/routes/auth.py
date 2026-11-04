"""Authentication and user administration endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from application.identity import (
    AuthService,
    ChangeRoleCommand,
    CreateUserCommand,
    LoginCommand,
    UserDTO,
    UserManagementService,
)
from application.ports.action_logger import ActionLoggerPort
from application.ports.rate_limit import RateLimitPolicyName
from domain.identity.roles import UserKind, UserRole
from presentation.api.auth_dependencies import get_current_user, require_admin
from presentation.api.dependencies import (
    create_action_logger,
    create_auth_service,
    create_user_management_service,
)
from presentation.api.rate_limit import rate_limit
from presentation.api.schemas import (
    ChangeRoleRequest,
    CreateUserRequest,
    CurrentUser,
    LoginRequest,
    MeResponse,
    TokenResponse,
    UserListResponse,
    UserResponse,
    UserToggleResponse,
)


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.LOGIN))],
)
async def login(
    *,
    req: LoginRequest,
    auth_service: Annotated[AuthService, Depends(create_auth_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> TokenResponse:
    result = await auth_service.authenticate(LoginCommand(email=req.email, password=req.password))
    log("login", details={"email": req.email})
    return TokenResponse(**result.__dict__)


@router.get("/me", response_model=MeResponse)
async def get_me(*, current_user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
    return current_user


@router.post(
    "/users",
    response_model=UserResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def add_user(
    *,
    req: CreateUserRequest,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    user_service: Annotated[UserManagementService, Depends(create_user_management_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> UserDTO:
    result = await user_service.create_user(
        CreateUserCommand(
            email=req.email,
            password=req.password,
            role=req.role or UserRole.USER,
            kind=req.kind or UserKind.INTERNAL,
        ),
        creator_role=admin.role,
    )
    log("user.create", user_id=admin.id, details={"email": req.email, "role": req.role})
    return result


@router.get("/users", response_model=UserListResponse)
async def list_all_users(
    *,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    user_service: Annotated[UserManagementService, Depends(create_user_management_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> UserListResponse:
    users, total = await user_service.list_users(limit=limit, offset=offset)
    log("user.list", user_id=admin.id, details={"limit": limit, "offset": offset})
    response_users = [
        UserResponse(id=u.id, email=u.email, role=u.role, kind=u.kind, is_active=u.is_active) for u in users
    ]
    return UserListResponse(total=total, limit=limit, offset=offset, users=response_users)


@router.patch(
    "/users/{user_id}",
    response_model=UserToggleResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def toggle_user_active(
    *,
    user_id: int,
    is_active: bool,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    user_service: Annotated[UserManagementService, Depends(create_user_management_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> UserToggleResponse:
    result = await user_service.set_active(user_id, is_active=is_active, admin_id=admin.id)
    log("user.toggle_active", user_id=admin.id, details={"target_user": user_id, "is_active": is_active})
    return UserToggleResponse(id=result.id, is_active=result.is_active)


@router.patch(
    "/users/{user_id}/role",
    response_model=UserResponse,
    dependencies=[Depends(rate_limit(RateLimitPolicyName.WRITE))],
)
async def change_user_role(
    *,
    user_id: int,
    body: ChangeRoleRequest,
    admin: Annotated[CurrentUser, Depends(require_admin)],
    user_service: Annotated[UserManagementService, Depends(create_user_management_service)],
    log: Annotated[ActionLoggerPort, Depends(create_action_logger)],
) -> UserDTO:
    result = await user_service.change_role(
        ChangeRoleCommand(user_id=user_id, new_role=body.role),
        admin_id=admin.id,
    )
    log(
        "user.change_role",
        user_id=admin.id,
        details={"target_user": user_id, "new_role": body.role},
    )
    return result
