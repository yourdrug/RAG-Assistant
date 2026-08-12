"""Chat-related DTOs -- immutable data-transfer objects for the chat API."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatResult:
    answer: str
    conversation_id: int
    sources: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class RagResult:
    """Structured result from the RAG pipeline with metadata for logging."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    breadth: str = "narrow"
    domain: str = "general"
    retrieval_count: int = 0
    reranker_score: float | None = None
    model_used: str | None = None
