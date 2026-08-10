"""StreamEvent -- tagged-union types for the RagService -> ChatService -> endpoint streaming protocol.

Replaces the fragile ``\\n__sources__`` / ``\\n__meta__`` string-sentinel
convention with a proper typed union (``TextChunk | SourcesEvent``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A piece of the LLM-generated answer text."""

    text: str


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    """Source metadata extracted from retrieved documents."""

    sources: list[dict]


@dataclass(frozen=True, slots=True)
class MetaEvent:
    """Final metadata: conversation_id and sources, yielded after the answer."""

    conversation_id: int
    sources: list[dict]


StreamEvent = TextChunk | SourcesEvent | MetaEvent
