"""
api/routes/auth.py — Auth endpoints. Thin wrappers around UserService.
"""

from fastapi import APIRouter, Depends
from infrastructure.auth import get_current_user, require_admin
from infrastructure.database import get_db
from services.user_service import UserService
from sqlalchemy.orm import Session

from api.schemas import CreateUserRequest, LoginRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    service = UserService()
    result = service.authenticate(req.email, req.password, db)
    return TokenResponse(**result)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user


@router.post("/users", response_model=UserResponse)
async def add_user(
    req: CreateUserRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    service = UserService()
    return service.create_user(req.email, req.password, req.role, req.kind, db)


@router.get("/users", response_model=list[UserResponse])
async def list_all_users(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    service = UserService()
    return service.list_users(db)


@router.patch("/users/{user_id}")
async def toggle_user_active(
    user_id: int,
    is_active: bool,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    service = UserService()
    return service.toggle_active(user_id, is_active, admin["id"], db)
