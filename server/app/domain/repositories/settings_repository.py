"""Settings Protocol — abstracts application settings for use cases."""

from __future__ import annotations

from typing import Protocol


class SettingsProtocol(Protocol):
    @property
    def history_window(self) -> int: ...
