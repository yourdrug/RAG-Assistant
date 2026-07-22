"""Auth infrastructure — backward-compatible re-exports.

The new DDD structure uses:
  - infrastructure.auth.password_hasher.BCryptPasswordHasher
  - infrastructure.auth.jwt_provider.JWTProvider
  - infrastructure.auth.fastapi_dependencies.get_current_user, require_admin

Legacy code imports:
  - infrastructure.auth.create_access_token, hash_password, verify_password, etc.
"""

from infrastructure.auth.fastapi_dependencies import get_current_user, require_admin
from infrastructure.auth.jwt_provider import JWTProvider
from infrastructure.auth.password_hasher import BCryptPasswordHasher

# Backward-compatible convenience functions for legacy code
_hasher = BCryptPasswordHasher()
_token_provider = JWTProvider()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _hasher.verify(password, hashed)


def create_access_token(user_id: int, role: str) -> str:
    return _token_provider.create_token(user_id, role)


def decode_access_token(token: str) -> dict:
    """Decode JWT token. Raises jwt exceptions on failure.

    HTTP mapping should be done in the presentation layer.
    """
    return _token_provider.decode_token(token)


__all__ = [
    "BCryptPasswordHasher",
    "JWTProvider",
    "get_current_user",
    "require_admin",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
]
