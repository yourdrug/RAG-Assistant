"""Auth endpoints — thin wrappers around AuthService."""

from __future__ import annotations

from application.dto.auth_dto import CreateUserCommand, LoginCommand
from application.services.auth_service import AuthService
from application.uow import UnitOfWork
from application.use_cases.auth.authenticate_user import AuthenticateUser
from application.use_cases.auth.create_user import CreateUser
from application.use_cases.auth.list_users import ListUsers
from application.use_cases.auth.toggle_user_active import ToggleUserActive
from fastapi import APIRouter, Depends
from infrastructure.auth.fastapi_dependencies import get_current_user, require_admin
from infrastructure.auth.jwt_provider import JWTProvider
from infrastructure.auth.password_hasher import BCryptPasswordHasher

from presentation.api.dependencies import get_uow
from presentation.api.schemas import CreateUserRequest, LoginRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])

_hasher = BCryptPasswordHasher()
_token_provider = JWTProvider()


def _auth_service(uow: UnitOfWork) -> AuthService:
    return AuthService(
        authenticate_user=AuthenticateUser(
            user_repo=uow.users,
            password_verifier=_hasher,
            token_provider=_token_provider,
        ),
        create_user=CreateUser(
            user_repo=uow.users,
            password_hasher=_hasher,
        ),
        list_users=ListUsers(user_repo=uow.users),
        toggle_user_active=ToggleUserActive(user_repo=uow.users),
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, uow: UnitOfWork = Depends(get_uow)):
    service = _auth_service(uow)
    result = service.authenticate(LoginCommand(email=req.email, password=req.password))
    return TokenResponse(**result.__dict__)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user


@router.post("/users", response_model=UserResponse)
async def add_user(
    req: CreateUserRequest,
    admin: dict = Depends(require_admin),
    uow: UnitOfWork = Depends(get_uow),
):
    service = _auth_service(uow)
    result = service.create_user(
        CreateUserCommand(email=req.email, password=req.password, role=req.role, kind=req.kind),
        creator_role=admin["role"],
    )
    return result


@router.get("/users", response_model=list[UserResponse])
async def list_all_users(
    admin: dict = Depends(require_admin),
    uow: UnitOfWork = Depends(get_uow),
):
    service = _auth_service(uow)
    return service.list_users()


@router.patch("/users/{user_id}")
async def toggle_user_active(
    user_id: int,
    is_active: bool,
    admin: dict = Depends(require_admin),
    uow: UnitOfWork = Depends(get_uow),
):
    service = _auth_service(uow)
    return service.toggle_active(user_id, is_active, admin["id"])
