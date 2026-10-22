"""Values passed between RAG stages; external client types stay in adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict


if TYPE_CHECKING:
    from domain.knowledge.chunk import Chunk
    from domain.rag.breadth import Breadth
    from domain.rag.context import RagContext
    from domain.rag.evidence.assessment import EvidenceAssessment
    from domain.rag.settings import RagSettings
    from domain.rag.stream_events import StreamEvent, UsageReport


@dataclass(frozen=True)
class PromptMessage:
    role: Literal["system", "human", "ai"]
    content: str


@dataclass(frozen=True)
class RagRequest:
    question: str
    history: tuple[PromptMessage, ...]
    ctx: RagContext
    settings: RagSettings
    started: float
    request_id: str
    model: str


@dataclass(frozen=True)
class PreparedQuery:
    request: RagRequest
    query: str
    q_hash: str = ""
    vis_hash: str = ""
    cache_revision: str | None = None


@dataclass(frozen=True)
class ScoredDocument:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    prepared: PreparedQuery
    breadth: Breadth
    domain: str
    fetch_k: int
    rerank_top_n: int
    dense_weight: float
    sparse_weight: float
    exact_ref_boost: bool
    candidates: tuple[Chunk, ...]
    documents: tuple[ScoredDocument, ...]
    sub_queries: tuple[str, ...]
    enumerate_cases: bool
    average_similarity: float


@dataclass(frozen=True)
class ContextBudget:
    num_ctx: int
    num_predict: int
    input_tokens: int
    base_tokens: int


@dataclass(frozen=True)
class PreparedContext:
    retrieval: RetrievalResult
    messages: tuple[PromptMessage, ...]
    history: tuple[PromptMessage, ...]
    documents: tuple[ScoredDocument, ...]
    budget: ContextBudget
    grounded_answer: str | None
    evidence: EvidenceAssessment | None


@dataclass(frozen=True)
class SanitizedAnswer:
    context: PreparedContext
    text: str
    usage: UsageReport | None
    ttft_sec: float | None


class SourceMetadata(TypedDict, total=False):
    source: str
    pages: list[str]
    doc_title: str
    doc_type: str
    sections: list[str]
    document_id: int
    content_hashes: list[str]
    articles: list[str]
    max_score: float
    score: float
    edited: bool
    manual: bool
    edited_at: str
    act_number: str
    effective_from: str
    effective_to: str
    citation_id: int


@dataclass(frozen=True)
class CompletedAnswer:
    answer: SanitizedAnswer
    sources: tuple[SourceMetadata, ...]
    confidence: float


@dataclass(frozen=True)
class TerminalResponse:
    events: tuple[StreamEvent, ...]


@dataclass(frozen=True)
class Sufficiency:
    sufficient: bool
    refinement: str | None
