"""Auth endpoints — thin wrappers around AuthService."""

from __future__ import annotations

from application.dto.auth_dto import CreateUserCommand, LoginCommand
from application.services.auth_service import AuthService
from fastapi import APIRouter, Depends

from presentation.api.auth_dependencies import get_current_user, require_admin
from presentation.api.dependencies import create_auth_service
from presentation.api.schemas import CreateUserRequest, LoginRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, auth_service: AuthService = Depends(create_auth_service)):
    result = await auth_service.authenticate(LoginCommand(email=req.email, password=req.password))
    return TokenResponse(**result.__dict__)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user


@router.post("/users", response_model=UserResponse)
async def add_user(
    req: CreateUserRequest,
    admin: dict = Depends(require_admin),
    auth_service: AuthService = Depends(create_auth_service),
):
    return await auth_service.create_user(
        CreateUserCommand(email=req.email, password=req.password, role=req.role, kind=req.kind),
        creator_role=admin["role"],
    )


@router.get("/users", response_model=list[UserResponse])
async def list_all_users(
    admin: dict = Depends(require_admin),
    auth_service: AuthService = Depends(create_auth_service),
):
    return await auth_service.list_users()


@router.patch("/users/{user_id}")
async def toggle_user_active(
    user_id: int,
    is_active: bool,
    admin: dict = Depends(require_admin),
    auth_service: AuthService = Depends(create_auth_service),
):
    return await auth_service.toggle_active(user_id, is_active, admin["id"])
