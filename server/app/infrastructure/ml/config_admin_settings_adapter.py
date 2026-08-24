"""Live adapter for ConfigAdminSettingsPort -- reads from global settings at access time."""

from __future__ import annotations

from config import settings


class LiveConfigAdminSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def llm_provider(self) -> str:  # type: ignore[override]
        return settings.llm_provider

    @property
    def llm_model(self) -> str:  # type: ignore[override]
        return settings.llm_model

    @property
    def tei_embed_url(self) -> str:  # type: ignore[override]
        return settings.tei_embed_url

    @property
    def tei_rerank_url(self) -> str:  # type: ignore[override]
        return settings.tei_rerank_url

    @property
    def ocr_engine(self) -> str:  # type: ignore[override]
        return settings.ocr_engine

    @property
    def ocr_enabled(self) -> bool:  # type: ignore[override]
        return settings.ocr_enabled

    @property
    def openrouter_model(self) -> str:  # type: ignore[override]
        return settings.openrouter_model

    @property
    def collection_name(self) -> str:  # type: ignore[override]
        return settings.collection_name
