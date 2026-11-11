"""JWT token provider -- creates and verifies PyJWT access tokens.

Tokens are signed with ``jwt_algorithm`` using ``jwt_secret_key`` from the injected
``AuthSection`` and expire after ``jwt_expire_minutes``. The section is read on every
call, so hot-reloaded values apply immediately. ``decode_token`` returns verified
``TokenClaims`` or raises ``jwt.InvalidTokenError`` (including for incomplete claims).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import jwt

from domain.identity.ports import TokenClaims


if TYPE_CHECKING:
    from config import AuthSection


REQUIRED_CLAIMS = ("sub", "role", "exp")


class JWTProvider:
    def __init__(self, auth_settings: AuthSection) -> None:
        self._settings = auth_settings

    def create_token(self, user_id: int, role: str) -> str:
        auth_settings = self._settings

        now = datetime.now(UTC)
        payload = {
            "sub": str(user_id),
            "role": role,
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(minutes=auth_settings.jwt_expire_minutes),
        }
        return jwt.encode(
            payload,
            auth_settings.jwt_secret_key,
            algorithm=auth_settings.jwt_algorithm,
        )

    def decode_token(self, token: str) -> TokenClaims:
        auth_settings = self._settings

        payload: object = jwt.decode(
            token,
            auth_settings.jwt_secret_key,
            algorithms=[auth_settings.jwt_algorithm],
            options={"require": list(REQUIRED_CLAIMS)},
        )
        return parse_claims(payload)


def parse_claims(payload: object) -> TokenClaims:
    """Validate the shape of a signature-verified payload."""
    if not isinstance(payload, dict):
        message = "Token payload must be an object"
        raise jwt.InvalidTokenError(message)

    subject: object = payload.get("sub")
    role: object = payload.get("role")
    expires: object = payload.get("exp")

    if not isinstance(subject, str) or not subject.isdecimal():
        message = "Token subject must be a numeric user id"
        raise jwt.InvalidTokenError(message)

    if not isinstance(role, str) or not role:
        message = "Token role must be a non-empty string"
        raise jwt.InvalidTokenError(message)

    if isinstance(expires, bool) or not isinstance(expires, int | float):
        message = "Token expiry must be a numeric timestamp"
        raise jwt.InvalidTokenError(message)

    return TokenClaims(
        user_id=int(subject),
        role=role,
        expires_at=datetime.fromtimestamp(expires, UTC),
    )
