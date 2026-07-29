"""FastAPI dependencies for authentication — presentation layer concern.

Поддерживает две независимые схемы в заголовке Authorization:
  - "Bearer <jwt>"    — internal-пользователи, как и раньше (JWT).
  - "Api-Key <key>"   — ТОЛЬКО внешние (kind='client') пользователи, статический ключ.
"""

from __future__ import annotations

import jwt as _jwt
from fastapi import Depends, HTTPException, status

from application.uow import UnitOfWork
from infrastructure.auth.api_key_provider import api_key_provider
from infrastructure.auth.jwt_provider import JWTProvider
from presentation.api.dependencies import get_uow, auth_key_header

jwt_provider = JWTProvider()


def _authenticate_via_jwt(token: str, uow: UnitOfWork) -> dict:
    try:
        payload = jwt_provider.decode_token(token)
    except _jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from None
    except _jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        ) from None

    user = uow.users.get_by_id(int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or deactivated"
        )

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "kind": user.kind,
        "is_active": user.is_active,
    }


def _authenticate_via_api_key(raw_key: str, uow: UnitOfWork) -> dict:
    key_hash = api_key_provider.hash_key(raw_key)
    cached = api_key_provider.get_cached(key_hash)

    if cached is api_key_provider.MISS:
        result = uow.api_keys.get_active_client_by_hash(key_hash)
        api_key_provider.set_cached(key_hash, result)
        if result is not None:
            uow.api_keys.touch_last_used(result["api_key_id"])
    else:
        result = cached

    if result is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key")

    return {
        "id": result["id"],
        "email": result["email"],
        "role": result["role"],
        "kind": result["kind"],
        "is_active": result["is_active"],
    }


def _parse_auth_header_value(value: str) -> tuple[str, str] | None:
    """
    Разбирает строку вида:
        Bearer <jwt>
        api-key <key>
    """
    scheme, _, credentials = value.strip().partition(" ")

    if not scheme or not credentials:
        return None

    return scheme, credentials


def get_current_user(
        authorization: str | None = Depends(auth_key_header),
        uow: UnitOfWork = Depends(get_uow),
) -> dict:
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parsed = _parse_auth_header_value(authorization)
    if parsed is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, credentials = parsed

    if scheme.lower() == "bearer":
        return _authenticate_via_jwt(credentials, uow)

    if scheme.lower() == "api-key":
        return _authenticate_via_api_key(credentials, uow)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unsupported authorization scheme",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin rights required")
    return current_user
