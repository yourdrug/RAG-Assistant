"""Redis connection settings."""

from __future__ import annotations

from urllib.parse import quote

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, SecretStr

from config.ownership import Secret, Static
from config.sections.database import MAX_TCP_PORT


class RedisSection(BaseModel):
    redis_host: Static[str] = "redis"
    redis_port: Static[int] = Field(default=6379, ge=1, le=MAX_TCP_PORT)
    redis_user: Static[str] = ""
    redis_password: Secret = SecretStr("")
    redis_db: Static[NonNegativeInt] = 0
    redis_socket_timeout: Static[PositiveInt] = 10
    redis_max_connections: Static[PositiveInt] = 50

    def check_redis_security(self) -> list[str]:
        if self.redis_password.get_secret_value() in ("", "password"):
            message = (
                "REDIS_PASSWORD must be a strong non-default value "
                "(set REDIS_PASSWORD in server/.env.secrets)"
            )
            return [message]

        return []

    @property
    def redis_url(self) -> str:
        # Percent-encode credentials: reserved characters would otherwise break DSN parsing.
        user = quote(self.redis_user, safe="")
        password = quote(self.redis_password.get_secret_value(), safe="")
        auth = ""

        if user:
            auth = user

            if password:
                auth = f"{auth}:{password}"

            auth = f"{auth}@"
        elif password:
            auth = f":{password}@"

        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
