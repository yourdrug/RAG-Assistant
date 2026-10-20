"""Runtime provider, storage, deployment and health constants."""

from __future__ import annotations

from enum import StrEnum


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class MLProvider(StrEnum):
    TEI = "tei"
    DEEPINFRA = "deepinfra"


class FileBackend(StrEnum):
    LOCAL = "local"
    S3 = "s3"


class AppStage(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PROD = "prod"


class HealthStatus(StrEnum):
    OK = "ok"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
