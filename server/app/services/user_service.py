"""
services/user_service.py — User authentication and management business logic.
"""

from fastapi import HTTPException
from infrastructure.auth import create_access_token, hash_password, verify_password
from infrastructure.database import (
    create_user,
    get_user_by_email,
    list_users,
    set_user_active,
)
from sqlalchemy.orm import Session


class UserService:
    def authenticate(self, email: str, password: str, db: Session) -> dict:
        user = get_user_by_email(db, email)
        if user is None or not user["is_active"] or not verify_password(password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = create_access_token(user_id=user["id"], role=user["role"])
        return {"access_token": token, "role": user["role"], "kind": user["kind"]}

    def create_user(
        self,
        email: str,
        password: str,
        role: str,
        kind: str,
        db: Session,
    ) -> dict:
        if role not in ("admin", "user"):
            raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'")
        if kind not in ("internal", "client"):
            raise HTTPException(status_code=400, detail="kind must be 'internal' or 'client'")
        if kind == "client" and role == "admin":
            raise HTTPException(status_code=400, detail="Client cannot be admin")
        if get_user_by_email(db, email) is not None:
            raise HTTPException(status_code=409, detail="User with this email already exists")

        return create_user(db, email=email, hashed_password=hash_password(password), role=role, kind=kind)

    def list_users(self, db: Session) -> list[dict]:
        return list_users(db)

    def toggle_active(self, user_id: int, is_active: bool, admin_id: int, db: Session) -> dict:
        if user_id == admin_id and not is_active:
            raise HTTPException(status_code=400, detail="Cannot deactivate yourself")
        if not set_user_active(db, user_id, is_active):
            raise HTTPException(status_code=404, detail="User not found")
        return {"id": user_id, "is_active": is_active}
