"""FastAPI dependencies for authentication — presentation layer concern.

Поддерживает две независимые схемы в заголовке Authorization:
  - "Bearer <jwt>"    — internal-пользователи, JWT.
  - "Api-Key <key>"   — ТОЛЬКО внешние (kind='client') пользователи, статический ключ.

Uses AuthService for user lookups — no direct UoW access.
"""

from __future__ import annotations

import jwt as _jwt
from application.services.auth_service import AuthService
from domain.exceptions import AuthenticationError, PermissionDeniedError
from domain.value_objects.capabilities import Capability, get_role_capabilities
from domain.value_objects.roles import UserRole
from fastapi import Depends
from fastapi.security import APIKeyHeader

from presentation.api.dependencies import create_api_key_provider, create_auth_service

auth_key_header = APIKeyHeader(
    name="Authorization",
    auto_error=False,
)


async def _authenticate_via_jwt(token: str, auth_service: AuthService) -> dict:
    try:
        payload = auth_service.decode_token(token)
    except _jwt.ExpiredSignatureError:
        raise AuthenticationError("Token expired") from None
    except _jwt.InvalidTokenError:
        raise AuthenticationError("Invalid or expired token") from None

    user = await auth_service.get_user_by_id(int(payload["sub"]))
    if user is None or not user.is_active:
        raise AuthenticationError("User not found or deactivated")

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "kind": user.kind,
        "is_active": user.is_active,
    }


async def _authenticate_via_api_key(raw_key: str, auth_service: AuthService, api_key_provider) -> dict:
    key_hash = api_key_provider.hash_key(raw_key)
    cached = await api_key_provider.get_cached(key_hash)

    if cached is api_key_provider.MISS:
        result = await auth_service.get_user_by_api_key_hash(key_hash)
        cache_value = result
        await api_key_provider.set_cached(key_hash, cache_value)
        if result is not None:
            await auth_service.touch_api_key_last_used(result.api_key_id)
    else:
        result = cached

    if result is None:
        raise AuthenticationError("Invalid or revoked API key")

    return {
        "id": result.id,
        "email": result.email,
        "role": result.role,
        "kind": result.kind,
        "is_active": result.is_active,
        "api_key_id": result.api_key_id,
    }


def _parse_auth_header_value(value: str) -> tuple[str, str] | None:
    scheme, _, credentials = value.strip().partition(" ")
    if not scheme or not credentials:
        return None
    return scheme, credentials


async def get_current_user(
    authorization: str | None = Depends(auth_key_header),
    auth_service: AuthService = Depends(create_auth_service),
    api_key_provider=Depends(create_api_key_provider),
) -> dict:
    if authorization is None:
        raise AuthenticationError("Not authenticated")

    parsed = _parse_auth_header_value(authorization)
    if parsed is None:
        raise AuthenticationError("Invalid Authorization header")

    scheme, credentials = parsed

    if scheme.lower() == "bearer":
        return await _authenticate_via_jwt(credentials, auth_service)

    if scheme.lower() == "api-key":
        return await _authenticate_via_api_key(credentials, auth_service, api_key_provider)

    raise AuthenticationError("Unsupported authorization scheme")


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user["role"] != UserRole.ADMIN:
        raise PermissionDeniedError("admin")
    return current_user


def require_capability(cap: Capability):
    """Return a dependency that checks if the current user has the given capability."""

    def _dep(current_user: dict = Depends(get_current_user)) -> dict:
        role = UserRole(current_user["role"])
        if cap not in get_role_capabilities(role):
            raise PermissionDeniedError(cap.value)
        return current_user

    return _dep
