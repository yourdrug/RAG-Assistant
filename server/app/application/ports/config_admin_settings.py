"""Config admin settings protocol -- abstracts dynamic model/OCR configuration.

Provides read-only access to admin-related settings that can be hot-reloaded
via ``/admin/config`` without a process restart.
"""

from __future__ import annotations

from typing import Protocol


class ConfigAdminSettingsPort(Protocol):
    @property
    def llm_provider(self) -> str: ...

    @property
    def llm_model(self) -> str: ...

    @property
    def tei_embed_url(self) -> str: ...

    @property
    def tei_rerank_url(self) -> str: ...

    @property
    def ocr_engine(self) -> str: ...

    @property
    def ocr_enabled(self) -> bool: ...

    @property
    def openrouter_model(self) -> str: ...

    @property
    def collection_name(self) -> str: ...
