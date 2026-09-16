"""Typed config dataclasses for presentation routes.

Routes depend on these instead of importing ``config.settings`` directly,
keeping the presentation layer decoupled from infrastructure configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UploadConfig:
    """Config for file upload validation in document/ingest routes."""

    max_upload_size_mb: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Config for benchmark execution routes."""

    data_dir: str
    retriever_top_k: int
    llm_model: str


@dataclass(frozen=True)
class CacheConfig:
    """Config for cache invalidation in document routes."""

    cache_enabled: bool


@dataclass(frozen=True)
class StorageConfig:
    """Config for storage-backend-dependent features (page images, Redis)."""

    file_backend: str
    redis_url: str
