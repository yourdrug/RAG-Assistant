"""Redis connection settings."""

from __future__ import annotations

from pydantic import BaseModel

from config.ownership import Secret, Static


class RedisSection(BaseModel):
    redis_host: Static[str] = "redis"
    redis_port: Static[int] = 6379
    redis_user: Static[str] = ""
    redis_password: Secret[str] = ""
    redis_db: Static[int] = 0
    redis_socket_timeout: Static[int] = 10
    redis_max_connections: Static[int] = 50

    def check_redis_security(self) -> list[str]:
        if self.redis_password in ("", "password"):
            message = (
                "REDIS_PASSWORD must be a strong non-default value "
                "(set REDIS_PASSWORD in server/.env.secrets)"
            )
            return [message]

        return []

    @property
    def redis_url(self) -> str:
        auth = ""

        if self.redis_user:
            auth = self.redis_user

            if self.redis_password:
                auth = f"{auth}:{self.redis_password}"

            auth = f"{auth}@"
        elif self.redis_password:
            auth = f":{self.redis_password}@"

        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
