"""Application configuration — Pydantic BaseSettings with .env support.

Fields are declared in ``config.sections``; ``Settings`` composes them into one
flat model so env names, dynamic DB parameter keys and hot-reload stay unchanged.
Deployment values come from env and supported dynamic values are applied from the
database at bootstrap. Target ownership is declared in application.runtime.setting_keys.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

from pydantic import model_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

from application.runtime.dynamic_params import DYNAMIC_PARAMS
from application.runtime.enums import AppStage
from config.sections import (
    AppSection,
    AuthSection,
    BenchmarkSection,
    ChatSection,
    ChunkingSection,
    ConcurrencySection,
    CostLimitSection,
    DatabaseSection,
    HttpSection,
    HybridSearchSection,
    JobsSection,
    LLMSection,
    MLProviderSection,
    OCRSection,
    RagFeaturesSection,
    RankingSection,
    RateLimitSection,
    RedisSection,
    RetrievalSection,
    StorageSection,
    VectorStoreSection,
)


if TYPE_CHECKING:
    from pydantic_settings import PydanticBaseSettingsSource

SERVER_DIR = Path(__file__).resolve().parent.parent.parent


def read_version() -> str:
    version_file = SERVER_DIR.parent / "VERSION"
    try:
        return version_file.read_text().strip()
    except (FileNotFoundError, OSError):
        return "0.0.0"


class EnvSettingsBase(BaseSettings):
    """Env-file configuration, declared on a single-inheritance base.

    Declaring it on ``Settings`` would override the plain ``ConfigDict`` that every
    section mixin inherits from ``BaseModel``; sections add no config of their own.
    """

    model_config = SettingsConfigDict(
        env_file=(
            str(SERVER_DIR / ".env"),
            str(SERVER_DIR / ".env.secrets"),
        ),
        extra="ignore",
    )


class Settings(
    EnvSettingsBase,
    AppSection,
    DatabaseSection,
    VectorStoreSection,
    LLMSection,
    MLProviderSection,
    ConcurrencySection,
    OCRSection,
    RetrievalSection,
    RankingSection,
    HybridSearchSection,
    RagFeaturesSection,
    ChunkingSection,
    ChatSection,
    RedisSection,
    JobsSection,
    StorageSection,
    HttpSection,
    AuthSection,
    RateLimitSection,
    CostLimitSection,
    BenchmarkSection,
):
    @classmethod
    def settings_customise_sources(
        cls,
        _settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Filter before decoding (legacy list values can contain invalid JSON).
        dynamic_keys = {param.key for param in DYNAMIC_PARAMS}

        for source in (env_settings, dotenv_settings):
            if not isinstance(source, EnvSettingsSource):
                continue

            source.env_vars = {
                key: value for key, value in source.env_vars.items() if key.lower() not in dynamic_keys
            }

        return init_settings, env_settings, dotenv_settings, file_secret_settings

    @model_validator(mode="after")
    def post_init(self) -> Settings:
        pool_errors = self.check_http_pools()

        if pool_errors:
            raise ValueError("; ".join(pool_errors))

        if not self.version:
            self.version = read_version()

        # Security violations are hard errors in EVERY stage: weak/blank
        # credentials must never reach any environment without warning.
        security_errors = self.check_security()

        if security_errors:
            msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in security_errors)
            sys.stderr.write(msg + "\n")
            sys.exit(1)

        return self

    def check_security(self) -> list[str]:
        # Credential hygiene is enforced in ALL stages (weak defaults never pass);
        # each section checks its own secrets, the origin rule spans App and Http.
        errors = [*self.check_auth_security(), *self.check_redis_security(), *self.check_storage_security()]

        if self.stage == AppStage.PROD and self.allowed_origins.strip() == "*":
            errors.append("ALLOWED_ORIGINS must not be '*' in production — list explicit origins")

        return errors


__all__ = ["Settings", "read_version"]
