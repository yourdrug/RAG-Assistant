"""JWT token provider -- creates and decodes PyJWT access tokens.

Tokens are signed with HS256 using the secret from ``settings.jwt_secret_key``
and expire after ``settings.jwt_expire_minutes``.  The ``decode_token`` method
returns the payload dict or raises ``jwt.InvalidTokenError``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from config import settings


class JWTProvider:
    def create_token(self, user_id: int, role: str) -> str:
        expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
        payload = {"sub": str(user_id), "role": role, "exp": expire}
        return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    def decode_token(self, token: str) -> dict:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
