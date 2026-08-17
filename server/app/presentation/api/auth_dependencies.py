"""FastAPI dependencies for authentication — presentation layer concern.

Поддерживает две независимые схемы в заголовке Authorization:
  - "Bearer <jwt>"    — internal-пользователи, как и раньше (JWT).
  - "Api-Key <key>"   — ТОЛЬКО внешние (kind='client') пользователи, статический ключ.
"""

from __future__ import annotations

import jwt as _jwt
from application.uow import UnitOfWork
from domain.exceptions import AuthenticationError, PermissionDeniedError
from domain.value_objects.roles import UserRole
from fastapi import Depends
from infrastructure.auth.api_key_provider import api_key_provider
from infrastructure.auth.jwt_provider import JWTProvider

from presentation.api.dependencies import auth_key_header, get_uow

jwt_provider = JWTProvider()


async def _authenticate_via_jwt(token: str, uow: UnitOfWork) -> dict:
    try:
        payload = jwt_provider.decode_token(token)
    except _jwt.ExpiredSignatureError:
        raise AuthenticationError("Token expired") from None
    except _jwt.InvalidTokenError:
        raise AuthenticationError("Invalid or expired token") from None

    user = await uow.users.get_by_id(int(payload["sub"]))
    if user is None or not user.is_active:
        raise AuthenticationError("User not found or deactivated")

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "kind": user.kind,
        "is_active": user.is_active,
    }


async def _authenticate_via_api_key(raw_key: str, uow: UnitOfWork) -> dict:
    key_hash = api_key_provider.hash_key(raw_key)
    cached = await api_key_provider.get_cached(key_hash)

    if cached is api_key_provider.MISS:
        result = await uow.api_keys.get_active_client_by_hash(key_hash)
        # Cache as dict for Redis serialization; repository returns typed value object
        cache_value = (
            {
                "api_key_id": result.api_key_id,
                "id": result.id,
                "email": result.email,
                "role": result.role,
                "kind": result.kind,
                "is_active": result.is_active,
            }
            if result is not None
            else None
        )
        await api_key_provider.set_cached(key_hash, cache_value)
        if result is not None:
            await uow.api_keys.touch_last_used(result.api_key_id)
    else:
        result = cached

    if result is None:
        raise AuthenticationError("Invalid or revoked API key")

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


async def get_current_user(
    authorization: str | None = Depends(auth_key_header),
    uow: UnitOfWork = Depends(get_uow),
) -> dict:
    if authorization is None:
        raise AuthenticationError("Not authenticated")

    parsed = _parse_auth_header_value(authorization)
    if parsed is None:
        raise AuthenticationError("Invalid Authorization header")

    scheme, credentials = parsed

    if scheme.lower() == "bearer":
        return await _authenticate_via_jwt(credentials, uow)

    if scheme.lower() == "api-key":
        return await _authenticate_via_api_key(credentials, uow)

    raise AuthenticationError("Unsupported authorization scheme")


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != UserRole.ADMIN:
        raise PermissionDeniedError("admin")
    return current_user
