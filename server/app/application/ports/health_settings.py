"""Health settings protocol -- abstracts dynamic health configuration.

Provides read-only access to health-related settings that can be hot-reloaded
via ``/admin/config`` without a process restart.
"""

from __future__ import annotations

from typing import Protocol


class HealthSettingsPort(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def uptime_seconds(self) -> float: ...

    @property
    def llm_provider(self) -> str: ...
