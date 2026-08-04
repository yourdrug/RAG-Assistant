"""Chat-related DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatResult:
    answer: str
    conversation_id: int
    sources: list[dict] = field(default_factory=list)
