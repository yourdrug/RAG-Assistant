"""JWT and bootstrap admin credentials."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, PositiveInt, SecretStr

from config.ownership import Secret, Static


# Shipped placeholder values that must never reach a running deployment.
PLACEHOLDER_SIGNING_KEYS = frozenset({"", "change-me-in-production"})
PLACEHOLDER_ADMIN_LOGINS = frozenset({"", "admin", "password", "change-me-in-production"})
# Only HMAC algorithms: the signing key is one shared secret. RFC 7518 §3.2 requires a key
# of at least the hash size; shorter keys can be brute-forced offline.
JWTAlgorithm = Literal["HS256", "HS384", "HS512"]
MIN_SIGNING_KEY_BYTES: dict[str, int] = {"HS256": 32, "HS384": 48, "HS512": 64}


class AuthSection(BaseModel):
    jwt_secret_key: Secret
    jwt_algorithm: Static[JWTAlgorithm] = "HS256"
    jwt_expire_minutes: Static[PositiveInt] = 1440
    admin_email: Static[str] = ""
    admin_password: Secret = SecretStr("")

    def check_auth_security(self) -> list[str]:
        errors: list[str] = []
        min_bytes = MIN_SIGNING_KEY_BYTES[self.jwt_algorithm]

        if (
            self.jwt_secret_key.get_secret_value() in PLACEHOLDER_SIGNING_KEYS
            or len(self.jwt_secret_key.get_secret_value().encode()) < min_bytes
        ):
            errors.append(
                f"JWT_SECRET_KEY must be a strong random value of at least {min_bytes} bytes "
                f"for {self.jwt_algorithm} (openssl rand -hex {min_bytes}) in server/.env.secrets"
            )

        if self.admin_email and self.admin_password.get_secret_value() in PLACEHOLDER_ADMIN_LOGINS:
            errors.append(
                "ADMIN_PASSWORD must be a strong non-default value "
                "(set ADMIN_PASSWORD in server/.env.secrets)"
            )

        return errors
