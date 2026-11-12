"""Application configuration package.

``Settings`` is one flat model composed of sections (``config.sections``). Modules
outside composition/configuration should depend on the section they need:

* ``request_section(Section)`` — the admitted request generation (falls back to
  the process singleton), i.e. the replacement for ``get_request_settings()``;
* ``process_section(Section)`` — the process singleton, i.e. the replacement for
  reading ``settings`` directly.

Both return the same live object narrowed to the section type, so hot-reload and
request snapshots behave exactly as before. The narrowing is static; at runtime
they only reject a class that is not a section.

Runtime helpers live in this module on purpose: they resolve ``settings`` from the
package namespace at call time, so ``patch("config.settings")`` keeps working.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Literal, cast, overload

from pydantic import BaseModel, ValidationError

from config.sections import (
    ALL_SECTIONS,
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
from config.settings import ConfigurationError, Settings, format_errors, read_version
from config.views import EmbeddingIdentitySettings, EndpointSettings, MLClientSettings


SECTION_TYPES: frozenset[type[BaseModel]] = frozenset(ALL_SECTIONS)


settings_overrides: ContextVar[dict[str, object] | None] = ContextVar("settings_overrides", default=None)

request_settings: ContextVar[Settings | None] = ContextVar("request_settings", default=None)


@overload
def get_setting(key: Literal["retriever_fetch_k", "retriever_top_k"]) -> int: ...


@overload
def get_setting(key: str) -> object: ...


def get_setting(key: str) -> object:
    """Read a setting, preferring per-context overrides over the global singleton.

    Used by benchmark/sweep code paths that need isolated config without
    mutating the process-wide ``settings`` object.
    """
    overrides = settings_overrides.get()

    if overrides is not None and key in overrides:
        return overrides[key]

    value: object = getattr(get_request_settings(), key)
    return value


def get_request_settings() -> Settings:
    """Use the admitted request generation until the operation releases it."""
    snapshot = request_settings.get()
    return snapshot if snapshot is not None else settings


def checked_section(section: type[object]) -> None:
    """Reject a class that is not one of the ``Settings`` sections.

    The check is on the argument, not on the returned object: tests substitute
    ``config.settings`` with mocks, and the object is ``Settings`` by construction.
    """
    if section not in SECTION_TYPES:
        message = f"{section.__name__} is not a Settings section"
        raise TypeError(message)


def request_section[SectionT: BaseModel](section: type[SectionT]) -> SectionT:
    """Request-generation settings narrowed to one section.

    Static narrowing only: at runtime this is the full ``Settings`` snapshot (which
    keeps hot reload working), not an isolated section object. ``Settings`` inherits
    every section in ``ALL_SECTIONS``, which makes the cast sound.
    """
    checked_section(section)
    return cast("SectionT", get_request_settings())


def process_section[SectionT: BaseModel](section: type[SectionT]) -> SectionT:
    """Process-wide settings narrowed to one section.

    Static narrowing only: at runtime this is the full ``Settings`` singleton.
    """
    checked_section(section)
    return cast("SectionT", settings)


def ml_client_settings() -> MLClientSettings:
    """Process-wide settings as the ML-client snapshot view."""
    return settings


def endpoint_settings() -> EndpointSettings:
    """Process-wide settings as the deployment-endpoint view."""
    return settings


def load_settings() -> Settings:
    """Build the process settings; every invalid-config failure surfaces as ``ConfigurationError``.

    Field errors are re-raised without their pydantic cause: it carries ``input_value``,
    which may be a credential.
    """
    try:
        return Settings()
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc']).upper() or 'SETTINGS'}: {error['msg']}"
            for error in exc.errors(include_url=False, include_input=False)
        ]
        raise ConfigurationError(format_errors(problems)) from None


settings = load_settings()

__all__ = [
    "ALL_SECTIONS",
    "AppSection",
    "AuthSection",
    "BenchmarkSection",
    "ChatSection",
    "ChunkingSection",
    "ConcurrencySection",
    "ConfigurationError",
    "CostLimitSection",
    "DatabaseSection",
    "EmbeddingIdentitySettings",
    "EndpointSettings",
    "HttpSection",
    "HybridSearchSection",
    "JobsSection",
    "LLMSection",
    "MLClientSettings",
    "MLProviderSection",
    "OCRSection",
    "RagFeaturesSection",
    "RankingSection",
    "RateLimitSection",
    "RedisSection",
    "RetrievalSection",
    "Settings",
    "StorageSection",
    "VectorStoreSection",
    "endpoint_settings",
    "get_request_settings",
    "get_setting",
    "load_settings",
    "ml_client_settings",
    "process_section",
    "read_version",
    "request_section",
    "request_settings",
    "settings",
    "settings_overrides",
]
