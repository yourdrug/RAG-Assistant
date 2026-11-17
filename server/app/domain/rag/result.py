"""RAG result — structured output from the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from domain.knowledge.enums import GENERAL_DOMAIN_KEY, DomainKey
from domain.rag.breadth import Breadth
from domain.rag.metadata import AnswerMetadata


if TYPE_CHECKING:
    from domain.rag.metadata import TokenUsage


@dataclass(frozen=True)
class RagResult(AnswerMetadata):
    """Structured result from the RAG pipeline with metadata for logging."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    breadth: Breadth = Breadth.NARROW
    domain: DomainKey = GENERAL_DOMAIN_KEY
    sources_count: int = 0
    config_data: dict[str, Any] | None = None
    aux_usage: TokenUsage | None = None
    confidence: float | None = None
    model_used: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    generation_sec: float | None = None
    ttft_sec: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", DomainKey(self.domain))
        object.__setattr__(self, "breadth", Breadth(self.breadth))
