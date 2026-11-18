"""Values passed between RAG stages; external client types stay in adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from domain.rag.outcome import AnswerOutcome


if TYPE_CHECKING:
    from application.rag.stream_events import StreamEvent, UsageReport
    from domain.knowledge.chunk import Chunk
    from domain.rag.breadth import Breadth
    from domain.rag.context import RagContext
    from domain.rag.evidence.assessment import EvidenceAssessment
    from domain.rag.settings import RagSettings


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
    generation_config: tuple[tuple[str, str | float], ...] = ()
    config_hash: str | None = None
    config_data: dict[str, Any] | None = None
    config_version: str | None = None
    evidence_profile: str | None = None
    usage_records: list[dict[str, Any]] = field(default_factory=list, compare=False)
    usage_start: int = 0
    trace_data: dict[str, Any] | None = field(default=None, compare=False)


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
    retrieval_ms: int | None = None
    rerank_ms: int | None = None
    refined_query: str | None = None
    top_rerank_score: float | None = None


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
    evidence_context: str = ""


@dataclass(frozen=True)
class SanitizedAnswer:
    context: PreparedContext
    text: str
    usage: UsageReport | None
    # Full answer production time (LLM stream + output scan/redaction), not time to first token.
    # ``text`` is authoritative: streamed chunks may stop early when the guardrails rewrite the answer.
    generation_sec: float | None
    # Time from generation start (including the LLM semaphore wait) to the first non-empty LLM chunk;
    # None when no LLM was called (grounded answer) or it produced no text.
    ttft_sec: float | None = None
    outcome: AnswerOutcome = AnswerOutcome.GENERATED


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
    outcome: AnswerOutcome = AnswerOutcome.NOT_FOUND
    retrieval: RetrievalResult | None = None


@dataclass(frozen=True)
class Sufficiency:
    sufficient: bool
    refinement: str | None
