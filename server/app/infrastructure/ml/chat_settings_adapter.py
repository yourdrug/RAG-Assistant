"""Live adapter for ChatSettingsPort -- reads from global settings at access time."""

from __future__ import annotations

from config import settings


class LiveChatSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def history_window(self) -> int:  # type: ignore[override]
        return settings.history_window

    @property
    def rolling_summary_enabled(self) -> bool:  # type: ignore[override]
        return settings.rolling_summary_enabled
