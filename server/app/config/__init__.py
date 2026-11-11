"""Application configuration package.

``Settings`` is one flat model composed of sections (``config.sections``). Modules
outside composition/configuration should depend on the section they need:

* ``request_section(Section)`` — the admitted request generation (falls back to
  the process singleton), i.e. the replacement for ``get_request_settings()``;
* ``process_section(Section)`` — the process singleton, i.e. the replacement for
  reading ``settings`` directly.

Both return the same live object narrowed to the section type, so hot-reload and
request snapshots behave exactly as before.

Runtime helpers live in this module on purpose: they resolve ``settings`` from the
package namespace at call time, so ``patch("config.settings")`` keeps working.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal, overload

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
from config.settings import Settings, read_version
from config.views import EmbeddingIdentitySettings, EndpointSettings, MLClientSettings


if TYPE_CHECKING:
    from pydantic import BaseModel


_settings_overrides: ContextVar[dict[str, object] | None] = ContextVar("settings_overrides", default=None)

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
    overrides = _settings_overrides.get()

    if overrides is not None and key in overrides:
        return overrides[key]

    value: object = getattr(get_request_settings(), key)
    return value


def get_request_settings() -> Settings:
    """Use the admitted request generation until the operation releases it."""
    snapshot = request_settings.get()
    return snapshot if snapshot is not None else settings


@overload
def request_section(_section: type[AppSection]) -> AppSection: ...


@overload
def request_section(_section: type[DatabaseSection]) -> DatabaseSection: ...


@overload
def request_section(_section: type[VectorStoreSection]) -> VectorStoreSection: ...


@overload
def request_section(_section: type[LLMSection]) -> LLMSection: ...


@overload
def request_section(_section: type[MLProviderSection]) -> MLProviderSection: ...


@overload
def request_section(_section: type[ConcurrencySection]) -> ConcurrencySection: ...


@overload
def request_section(_section: type[OCRSection]) -> OCRSection: ...


@overload
def request_section(_section: type[RetrievalSection]) -> RetrievalSection: ...


@overload
def request_section(_section: type[RankingSection]) -> RankingSection: ...


@overload
def request_section(_section: type[HybridSearchSection]) -> HybridSearchSection: ...


@overload
def request_section(_section: type[RagFeaturesSection]) -> RagFeaturesSection: ...


@overload
def request_section(_section: type[ChunkingSection]) -> ChunkingSection: ...


@overload
def request_section(_section: type[ChatSection]) -> ChatSection: ...


@overload
def request_section(_section: type[RedisSection]) -> RedisSection: ...


@overload
def request_section(_section: type[JobsSection]) -> JobsSection: ...


@overload
def request_section(_section: type[StorageSection]) -> StorageSection: ...


@overload
def request_section(_section: type[HttpSection]) -> HttpSection: ...


@overload
def request_section(_section: type[AuthSection]) -> AuthSection: ...


@overload
def request_section(_section: type[RateLimitSection]) -> RateLimitSection: ...


@overload
def request_section(_section: type[CostLimitSection]) -> CostLimitSection: ...


@overload
def request_section(_section: type[BenchmarkSection]) -> BenchmarkSection: ...


def request_section(_section: type[BaseModel]) -> BaseModel:
    """Request-generation settings narrowed to one section; the argument only selects the type.

    ``Settings`` inherits every section, so each overload is satisfied without a cast.
    """
    return get_request_settings()


@overload
def process_section(_section: type[AppSection]) -> AppSection: ...


@overload
def process_section(_section: type[DatabaseSection]) -> DatabaseSection: ...


@overload
def process_section(_section: type[VectorStoreSection]) -> VectorStoreSection: ...


@overload
def process_section(_section: type[LLMSection]) -> LLMSection: ...


@overload
def process_section(_section: type[MLProviderSection]) -> MLProviderSection: ...


@overload
def process_section(_section: type[ConcurrencySection]) -> ConcurrencySection: ...


@overload
def process_section(_section: type[OCRSection]) -> OCRSection: ...


@overload
def process_section(_section: type[RetrievalSection]) -> RetrievalSection: ...


@overload
def process_section(_section: type[RankingSection]) -> RankingSection: ...


@overload
def process_section(_section: type[HybridSearchSection]) -> HybridSearchSection: ...


@overload
def process_section(_section: type[RagFeaturesSection]) -> RagFeaturesSection: ...


@overload
def process_section(_section: type[ChunkingSection]) -> ChunkingSection: ...


@overload
def process_section(_section: type[ChatSection]) -> ChatSection: ...


@overload
def process_section(_section: type[RedisSection]) -> RedisSection: ...


@overload
def process_section(_section: type[JobsSection]) -> JobsSection: ...


@overload
def process_section(_section: type[StorageSection]) -> StorageSection: ...


@overload
def process_section(_section: type[HttpSection]) -> HttpSection: ...


@overload
def process_section(_section: type[AuthSection]) -> AuthSection: ...


@overload
def process_section(_section: type[RateLimitSection]) -> RateLimitSection: ...


@overload
def process_section(_section: type[CostLimitSection]) -> CostLimitSection: ...


@overload
def process_section(_section: type[BenchmarkSection]) -> BenchmarkSection: ...


def process_section(_section: type[BaseModel]) -> BaseModel:
    """Process-wide settings narrowed to one section; the argument only selects the type.

    ``Settings`` inherits every section, so each overload is satisfied without a cast.
    """
    return settings


def ml_client_settings() -> MLClientSettings:
    """Process-wide settings as the ML-client snapshot view."""
    return settings


def endpoint_settings() -> EndpointSettings:
    """Process-wide settings as the deployment-endpoint view."""
    return settings


settings = Settings()

__all__ = [
    "ALL_SECTIONS",
    "AppSection",
    "AuthSection",
    "BenchmarkSection",
    "ChatSection",
    "ChunkingSection",
    "ConcurrencySection",
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
    "_settings_overrides",
    "endpoint_settings",
    "get_request_settings",
    "get_setting",
    "ml_client_settings",
    "process_section",
    "read_version",
    "request_section",
    "request_settings",
    "settings",
]
