"""JWT and bootstrap admin credentials."""

from __future__ import annotations

from pydantic import BaseModel

from config.ownership import Secret, Static


# Shipped placeholder values that must never reach a running deployment.
PLACEHOLDER_SIGNING_KEYS = frozenset({"", "change-me-in-production"})
PLACEHOLDER_ADMIN_LOGINS = frozenset({"", "admin", "password", "change-me-in-production"})


class AuthSection(BaseModel):
    jwt_secret_key: Secret[str]
    jwt_algorithm: Static[str] = "HS256"
    jwt_expire_minutes: Static[int] = 1440
    admin_email: Static[str] = ""
    admin_password: Secret[str] = ""

    def check_auth_security(self) -> list[str]:
        errors: list[str] = []

        if self.jwt_secret_key in PLACEHOLDER_SIGNING_KEYS:
            errors.append(
                "JWT_SECRET_KEY must be set to a strong random value "
                "(openssl rand -hex 32) in server/.env.secrets"
            )

        if self.admin_email and self.admin_password in PLACEHOLDER_ADMIN_LOGINS:
            errors.append(
                "ADMIN_PASSWORD must be a strong non-default value "
                "(set ADMIN_PASSWORD in server/.env.secrets)"
            )

        return errors
