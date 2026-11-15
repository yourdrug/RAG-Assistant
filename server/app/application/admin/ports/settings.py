"""Config admin settings protocol -- abstracts dynamic model/OCR configuration.

Provides read-only access to admin-related settings that can be hot-reloaded
via ``/admin/config`` without a process restart.
"""

from __future__ import annotations

from typing import Protocol


class ConfigAdminSettingsPort(Protocol):
    @property
    def llm_base_url(self) -> str: ...

    @property
    def llm_model(self) -> str: ...

    @property
    def llm_auxiliary_model(self) -> str: ...

    @property
    def llm_api_key_configured(self) -> bool: ...

    @property
    def embedding_base_url(self) -> str: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def embedding_api_key_configured(self) -> bool: ...

    @property
    def rerank_base_url(self) -> str: ...

    @property
    def rerank_model(self) -> str: ...

    @property
    def rerank_api_format(self) -> str: ...

    @property
    def rerank_api_key_configured(self) -> bool: ...

    @property
    def ocr_engine(self) -> str: ...

    @property
    def ocr_enabled(self) -> bool: ...

    @property
    def collection_name(self) -> str: ...
