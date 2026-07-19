import logging

from fastapi import APIRouter, Depends, HTTPException
from infrastructure.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)
from infrastructure.database import (
    create_user,
    get_db,
    get_user_by_email,
    list_users,
    set_user_active,
)
from sqlalchemy.orm import Session

from api.schemas import (
    CreateUserRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
)

logger = logging.getLogger("default")

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, req.email)
    if user is None or not user["is_active"] or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    token = create_access_token(user_id=user["id"], role=user["role"])
    return TokenResponse(access_token=token, role=user["role"], kind=user["kind"])


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user


@router.post("/users", response_model=UserResponse)
async def add_user(
    req: CreateUserRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role должен быть 'admin' или 'user'")
    if req.kind not in ("internal", "client"):
        raise HTTPException(status_code=400, detail="kind должен быть 'internal' или 'client'")
    if req.kind == "client" and req.role == "admin":
        raise HTTPException(status_code=400, detail="Клиент не может быть admin")
    if get_user_by_email(db, req.email) is not None:
        raise HTTPException(status_code=409, detail="Пользователь с таким email уже существует")

    user = create_user(
        db, email=req.email, hashed_password=hash_password(req.password), role=req.role, kind=req.kind
    )
    return user


@router.get("/users", response_model=list[UserResponse])
async def list_all_users(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return list_users(db)


@router.patch("/users/{user_id}")
async def toggle_user_active(
    user_id: int,
    is_active: bool,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if user_id == admin["id"] and not is_active:
        raise HTTPException(status_code=400, detail="Нельзя деактивировать самого себя")
    if not set_user_active(db, user_id, is_active):
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return {"id": user_id, "is_active": is_active}
