"""ConfigMaskerPort — masks sensitive config values for display/logging."""

from __future__ import annotations

from typing import Protocol


class ConfigMaskerPort(Protocol):
    """Masks sensitive configuration values."""

    @property
    def sensitive_keys(self) -> frozenset[str]: ...

    def mask_value(self, value: str | None) -> str: ...
