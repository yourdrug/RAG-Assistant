"""FastAPI dependencies for authentication — presentation layer concern.

Поддерживает две независимые схемы в заголовке Authorization:
  - "Bearer <jwt>"    — internal-пользователи, JWT.
  - "Api-Key <key>"   — ТОЛЬКО внешние (kind='client') пользователи, статический ключ.

Uses identity services for authentication lookups — no direct UoW access.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from fastapi import Depends
from fastapi.security import APIKeyHeader
import jwt as _jwt

from application.budget import billing_principal
from application.identity import ApiKeyProviderPort, ApiKeyService, AuthService
from domain.identity.capabilities import Capability, get_role_capabilities
from domain.identity.roles import UserRole
from domain.shared.exceptions import AuthenticationError, PermissionDeniedError
from presentation.api.constants import AUTH_SCHEME_API_KEY, AUTH_SCHEME_BEARER
from presentation.api.dependencies import create_api_key_provider, create_api_key_service, create_auth_service
from presentation.api.schemas import CurrentUser


auth_key_header = APIKeyHeader(
    name="Authorization",
    auto_error=False,
)


async def authenticate_via_jwt(token: str, auth_service: AuthService) -> CurrentUser:
    try:
        payload = auth_service.decode_token(token)
    except _jwt.ExpiredSignatureError:
        message = "Token expired"
        raise AuthenticationError(message) from None
    except _jwt.InvalidTokenError:
        message = "Invalid or expired token"
        raise AuthenticationError(message) from None

    user = await auth_service.get_user_by_id(int(payload["sub"]))

    if user is None or not user.is_active:
        message = "User not found or deactivated"
        raise AuthenticationError(message)

    return CurrentUser(
        id=user.id,
        email=user.email,
        role=user.role,
        kind=user.kind,
        is_active=user.is_active,
    )


async def authenticate_via_api_key(
    raw_key: str, api_key_service: ApiKeyService, api_key_provider: ApiKeyProviderPort
) -> CurrentUser:
    key_hash = api_key_provider.hash_key(raw_key)
    cached = await api_key_provider.get_cached(key_hash)

    if cached is not api_key_provider.MISS:
        if cached is None:
            message = "Invalid or revoked API key"
            raise AuthenticationError(message)

        return CurrentUser.model_validate(cached)

    result = await api_key_service.get_user_by_api_key_hash(key_hash)
    cache_value = asdict(result) if result is not None else None
    await api_key_provider.set_cached(key_hash, cache_value)

    if result is None:
        message = "Invalid or revoked API key"
        raise AuthenticationError(message)

    await api_key_service.touch_api_key_last_used(result.api_key_id)

    return CurrentUser(
        id=result.id,
        email=result.email,
        role=result.role,
        kind=result.kind,
        is_active=result.is_active,
        api_key_id=result.api_key_id,
    )


def parse_auth_header_value(value: str) -> tuple[str, str] | None:
    scheme, _, credentials = value.strip().partition(" ")

    if not scheme or not credentials:
        return None

    return scheme, credentials


async def get_current_user(
    authorization: str | None = Depends(auth_key_header),
    auth_service: AuthService = Depends(create_auth_service),
    api_key_provider: ApiKeyProviderPort = Depends(create_api_key_provider),
    api_key_service: ApiKeyService = Depends(create_api_key_service),
) -> CurrentUser:
    if authorization is None:
        message = "Not authenticated"
        raise AuthenticationError(message)

    parsed = parse_auth_header_value(authorization)

    if parsed is None:
        message = "Invalid Authorization header"
        raise AuthenticationError(message)

    scheme, credentials = parsed

    if scheme.lower() == AUTH_SCHEME_BEARER:
        user = await authenticate_via_jwt(credentials, auth_service)
        billing_principal.set(f"user:{user.id}")
        return user

    if scheme.lower() == AUTH_SCHEME_API_KEY:
        user = await authenticate_via_api_key(credentials, api_key_service, api_key_provider)
        billing_principal.set(f"user:{user.id}")
        return user

    message = "Unsupported authorization scheme"
    raise AuthenticationError(message)


def require_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if current_user.role != UserRole.ADMIN:
        message = "admin"
        raise PermissionDeniedError(message)

    return current_user


def require_capability(cap: Capability) -> Callable[..., CurrentUser]:
    """Return a dependency that checks if the current user has the given capability."""

    def check_capability(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        role = UserRole(current_user.role)

        if cap not in get_role_capabilities(role):
            raise PermissionDeniedError(cap.value)

        return current_user

    return check_capability
