"""Chat-related DTOs -- immutable data-transfer objects for the chat API."""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth


@dataclass(frozen=True)
class ChatResult:
    answer: str
    conversation_id: int
    sources: list[dict] = field(default_factory=list)
    confidence: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class RagResult:
    """Structured result from the RAG pipeline with metadata for logging."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    breadth: str = Breadth.NARROW.value
    domain: str = DocDomain.GENERAL.value
    retrieval_count: int = 0
    reranker_score: float | None = None
    confidence: float | None = None
    model_used: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
