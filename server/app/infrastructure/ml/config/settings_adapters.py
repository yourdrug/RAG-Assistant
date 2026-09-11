"""Live adapters for settings ports — reads from global settings singleton at access time.

Consolidates all four settings adapter classes into a single module.
"""

from __future__ import annotations

from config import settings


class LiveChunkSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def chunk_size(self) -> int:
        return settings.chunk_size


class LiveChatSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def history_window(self) -> int:
        return settings.history_window

    @property
    def rolling_summary_enabled(self) -> bool:
        return settings.rolling_summary_enabled


class LiveHealthSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def version(self) -> str:
        return settings.version

    @property
    def uptime_seconds(self) -> float:
        return settings.uptime_seconds

    @property
    def llm_provider(self) -> str:
        return settings.llm_provider


class LiveConfigAdminSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def llm_provider(self) -> str:
        return settings.llm_provider

    @property
    def llm_model(self) -> str:
        return settings.llm_model

    @property
    def tei_embed_url(self) -> str:
        return settings.tei_embed_url

    @property
    def tei_rerank_url(self) -> str:
        return settings.tei_rerank_url

    @property
    def ml_provider(self) -> str:
        return settings.ml_provider

    @property
    def deepinfra_embed_model(self) -> str:
        return settings.deepinfra_embed_model

    @property
    def deepinfra_rerank_model(self) -> str:
        return settings.deepinfra_rerank_model

    @property
    def ocr_engine(self) -> str:
        return settings.ocr_engine

    @property
    def ocr_enabled(self) -> bool:
        return settings.ocr_enabled

    @property
    def openrouter_model(self) -> str:
        return settings.openrouter_model

    @property
    def collection_name(self) -> str:
        return settings.collection_name
