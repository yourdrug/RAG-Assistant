"""Application configuration — Pydantic BaseSettings with .env support.

Fields are declared in ``config.sections``; ``Settings`` composes them into one
flat model so env names, dynamic DB parameter keys and hot-reload stay unchanged.
Deployment values come from env and supported dynamic values are applied from the
database at bootstrap. Target ownership is declared in application.runtime.setting_keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import dotenv_values
from pydantic import model_validator
from pydantic_settings import BaseSettings, DotEnvSettingsSource, EnvSettingsSource, SettingsConfigDict

from application.runtime.dynamic_params import DYNAMIC_PARAMS
from application.runtime.enums import AppStage
from application.runtime.errors import ConfigurationError
from config.ownership import SettingOwnership, keys_owned_by
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
SECRETS_FILE_NAME = ".env.secrets"


def format_errors(errors: list[str]) -> str:
    return "Configuration errors:\n" + "\n".join(f"  - {error}" for error in errors)


def misplaced_secrets(source: DotEnvSettingsSource, secret_keys: frozenset[str]) -> list[str]:
    """Secret keys found in a dotenv file other than ``.env.secrets`` (names only, never values)."""
    env_files = source.env_file

    if env_files is None:
        paths: list[Path] = []
    elif isinstance(env_files, (str, Path)):
        paths = [Path(env_files)]
    else:
        paths = [Path(item) for item in env_files]

    errors: list[str] = []

    for path in paths:
        resolved = path.expanduser()

        if resolved.name == SECRETS_FILE_NAME or not resolved.is_file():
            continue

        values = dotenv_values(resolved, encoding=source.env_file_encoding)
        errors.extend(
            f"{key.upper()} is a secret: set it in {SECRETS_FILE_NAME} or the process environment, "
            f"not in {resolved.name}"
            for key in sorted(values)
            if key.lower() in secret_keys
        )

    return errors


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
            str(SERVER_DIR / SECRETS_FILE_NAME),
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

        if isinstance(dotenv_settings, DotEnvSettingsSource):
            errors = misplaced_secrets(dotenv_settings, keys_owned_by(cls, SettingOwnership.SECRET))

            if errors:
                raise ConfigurationError(format_errors(errors))

        return init_settings, env_settings, dotenv_settings, file_secret_settings

    @model_validator(mode="after")
    def post_init(self) -> Settings:
        structure_errors = [
            *self.check_http_pools(),
            *self.check_database(),
            *self.check_retrieval(),
            *self.check_generation_budget(),
        ]

        if structure_errors:
            raise ConfigurationError(format_errors(structure_errors))

        if not self.version:
            self.version = read_version()

        # Security violations are hard errors in EVERY stage: weak/blank
        # credentials must never reach any environment without warning.
        security_errors = self.check_security()

        if security_errors:
            raise ConfigurationError(format_errors(security_errors))

        return self

    def check_security(self) -> list[str]:
        # Credential hygiene is enforced in ALL stages (weak defaults never pass);
        # each section checks its own secrets, the origin rule spans App and Http.
        errors = [*self.check_auth_security(), *self.check_redis_security(), *self.check_storage_security()]

        if self.stage == AppStage.PROD and "*" in self.allowed_origins_list:
            errors.append("ALLOWED_ORIGINS must not be '*' in production — list explicit origins")

        return errors


__all__ = ["ConfigurationError", "Settings", "format_errors", "read_version"]
