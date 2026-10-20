"""RAG result — structured output from the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.knowledge.enums import DocDomain
from domain.rag.breadth import Breadth


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
    ttft_sec: float | None = None
