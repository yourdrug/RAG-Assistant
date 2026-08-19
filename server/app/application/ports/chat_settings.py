"""Chat settings protocol -- abstracts dynamic chat configuration.

Provides read-only access to chat-related settings that can be hot-reloaded
via ``/admin/config`` without a process restart.
"""

from __future__ import annotations

from typing import Protocol


class ChatSettingsPort(Protocol):
    @property
    def history_window(self) -> int: ...

    @property
    def rolling_summary_enabled(self) -> bool: ...
