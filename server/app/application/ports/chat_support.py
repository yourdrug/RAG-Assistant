"""Chat support ports — abstract interfaces for chat-related infrastructure."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RollingSummaryUpdaterPort(Protocol):
    """Updates rolling summaries for conversations."""

    async def update(self, existing_summary: str | None, recent_turns: list[dict]) -> str: ...
