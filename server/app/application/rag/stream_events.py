r"""StreamEvent -- tagged-union types for the RagService -> ChatService -> endpoint streaming protocol.

Replaces the fragile ``\n__sources__`` / ``\n__meta__`` string-sentinel
convention with a proper typed union (``TextChunk | SourcesEvent``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from domain.rag.metadata import AnswerMetadata


if TYPE_CHECKING:
    from datetime import date

    from domain.rag.breadth import Breadth
    from domain.rag.metadata import TokenUsage


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A piece of the LLM-generated answer text."""

    text: str


@dataclass(frozen=True, slots=True)
class UsageReport:
    """LLM token usage and cost for the request."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""
    operation: str = "generate"


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    """Source metadata extracted from retrieved documents."""

    sources: list[dict[str, Any]]
    confidence: float | None = None
    usage: UsageReport | None = None
    # The final sanitized answer of a generated response; streamed chunks are only its prefix.
    answer: str | None = None


@dataclass(frozen=True, slots=True)
class StatusEvent:
    """Pipeline status update for the frontend progress indicator."""

    stage: str


@dataclass(frozen=True, slots=True)
class MetaEvent:
    """Final metadata: conversation_id and sources, yielded after the answer."""

    conversation_id: int
    sources: list[dict[str, Any]]
    confidence: float | None = None
    usage: UsageReport | None = None
    as_of_date: date | None = None
    message_id: int | None = None
    answer: str = ""


@dataclass(frozen=True, slots=True)
class PipelineMetaEvent(AnswerMetadata):
    """Pipeline metadata: breadth, domain, TTFT, timings.

    Internal snapshots are emitted before fallible stages and on completion,
    so ChatService can record the known metadata even on error or cancellation.
    The final snapshot includes the actual classification and terminal outcome.
    """

    breadth: Breadth | None = None
    domain: str = ""
    model_used: str | None = None
    config_data: dict[str, Any] | None = None
    trace_data: dict[str, Any] | None = None
    aux_usage: TokenUsage | None = None
    generation_sec: float | None = None
    ttft_sec: float | None = None


StreamEvent = TextChunk | SourcesEvent | StatusEvent | MetaEvent | PipelineMetaEvent
