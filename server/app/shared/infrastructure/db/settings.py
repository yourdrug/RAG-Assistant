"""Pydantic-settings for database configuration (KinTree-style)."""

from __future__ import annotations

from typing import ClassVar
from zoneinfo import ZoneInfo

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database — master
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str
    DB_NAME: str

    # Database — slaves (optional, for cluster mode)
    DB_SLAVE_HOSTS: list[str] | None = Field(default=None)
    DB_SLAVE_PORTS: list[str] | None = Field(default=None)

    # Timezone (IANA tz name)
    TIMEZONE: str = Field(default="UTC")

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore",
    )

    @field_validator("TIMEZONE")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except Exception as exc:
            raise ValueError(f"Invalid IANA timezone: {v!r}") from exc
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.TIMEZONE)


settings: Settings = Settings()
