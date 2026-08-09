"""MessageRole value object -- enum for conversation turn authorship."""

from __future__ import annotations

from enum import StrEnum


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
