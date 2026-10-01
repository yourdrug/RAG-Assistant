r"""StreamEvent -- tagged-union types for the RagService -> ChatService -> endpoint streaming protocol.

Replaces the fragile ``\n__sources__`` / ``\n__meta__`` string-sentinel
convention with a proper typed union (``TextChunk | SourcesEvent``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from domain.value_objects.llm_provider import Breadth


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A piece of the LLM-generated answer text."""

    text: str


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    """Source metadata extracted from retrieved documents."""

    sources: list[dict]
    confidence: float | None = None
    usage: UsageReport | None = None


@dataclass(frozen=True, slots=True)
class UsageReport:
    """LLM token usage and cost for the request."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""
    operation: str = "generate"


@dataclass(frozen=True, slots=True)
class StatusEvent:
    """Pipeline status update for the frontend progress indicator."""

    stage: str


@dataclass(frozen=True, slots=True)
class MetaEvent:
    """Final metadata: conversation_id and sources, yielded after the answer."""

    conversation_id: int
    sources: list[dict]
    confidence: float | None = None
    usage: UsageReport | None = None
    as_of_date: date | None = None


@dataclass(frozen=True, slots=True)
class PipelineMetaEvent:
    """Pipeline metadata: breadth, domain, TTFT, timings.

    Yielded once by RagService.stream() after pipeline completes, carries
    the actual breadth/domain classification that was used inside the
    pipeline (not the re-classified version on the original question).
    """

    breadth: Breadth | None = None
    domain: str = ""
    ttft_sec: float | None = None


StreamEvent = TextChunk | SourcesEvent | StatusEvent | MetaEvent | PipelineMetaEvent
