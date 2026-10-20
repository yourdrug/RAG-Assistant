"""Ports owned by chat orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from domain.chat.message import Message
    from domain.rag.context import RagContext
    from domain.rag.result import RagResult
    from domain.rag.stream_events import StreamEvent


@runtime_checkable
class ChatRAGPort(Protocol):
    def stream(
        self,
        question: str,
        history: list[Message],
        ctx: RagContext,
    ) -> AsyncIterator[StreamEvent]: ...

    async def invoke(
        self,
        question: str,
        history: list[Message],
        ctx: RagContext,
    ) -> RagResult: ...


class ChatSettingsPort(Protocol):
    @property
    def history_window(self) -> int: ...

    @property
    def rolling_summary_enabled(self) -> bool: ...

    @property
    def curator_scope_max_ids(self) -> int: ...


@runtime_checkable
class RollingSummaryUpdaterPort(Protocol):
    """Updates rolling summaries for conversations."""

    async def update(self, existing_summary: str | None, recent_turns: list[dict[str, Any]]) -> str: ...


@runtime_checkable
class PIIRedactorPort(Protocol):
    """Detect and redact PII in text. Handles enabled/disabled logic internally."""

    def redact(self, text: str) -> str: ...

    def scan_and_redact(self, text: str) -> tuple[str, list[str]]:
        """Scan for PII and return (redacted_text, list_of_detected_types)."""
        ...
