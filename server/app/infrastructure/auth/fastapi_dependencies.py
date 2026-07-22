"""FastAPI dependencies for authentication — presentation layer concern."""

from __future__ import annotations

import jwt as _jwt
from application.uow import UnitOfWork
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from presentation.api.dependencies import get_uow

from infrastructure.auth.jwt_provider import JWTProvider

bearer_scheme = HTTPBearer(auto_error=False)
jwt_provider = JWTProvider()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    uow: UnitOfWork = Depends(get_uow),
) -> dict:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt_provider.decode_token(credentials.credentials)
    except _jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        ) from None
    except _jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from None

    user = uow.users.get_by_id(int(payload["sub"]))

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
