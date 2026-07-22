"""FastAPI dependencies for authentication — presentation layer concern."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from infrastructure.auth.jwt_provider import JWTProvider
from infrastructure.database import get_db
from infrastructure.repositories.sqlalchemy_user_repository import SQLAlchemyUserRepository

bearer_scheme = HTTPBearer(auto_error=False)
jwt_provider = JWTProvider()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> dict:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt_provider.decode_token(credentials.credentials)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from None

    user_repo = SQLAlchemyUserRepository(db)
    user = user_repo.get_by_id(int(payload["sub"]))

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "kind": user.kind,
        "is_active": user.is_active,
    }


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin rights required",
        )
    return current_user
