"""Live adapter for HealthSettingsPort -- reads from global settings at access time."""

from __future__ import annotations

from config import settings


class LiveHealthSettings:
    """Each property read returns the current value from the global settings singleton."""

    @property
    def version(self) -> str:  # type: ignore[override]
        return settings.version

    @property
    def uptime_seconds(self) -> float:  # type: ignore[override]
        return settings.uptime_seconds

    @property
    def llm_provider(self) -> str:  # type: ignore[override]
        return settings.llm_provider
