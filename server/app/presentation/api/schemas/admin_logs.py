"""Admin logs & chat logs schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from domain.chat.review import MAX_REVIEW_NOTE_CHARS, ReviewStatus
from domain.rag.outcome import AnswerOutcome
from presentation.api.schemas.chat import FeedbackResponse


class LogEntry(BaseModel):
    timestamp: str
    level: str
    logger: str
    request_id: str
    message: str
    filename: str | None = None
    lineno: int | None = None
    user_id: int | None = None
    user_login: str | None = None


class LogsResponse(BaseModel):
    logs: list[LogEntry]
    total: int


class ChatLogReviewResponse(BaseModel):
    status: ReviewStatus
    note: str | None = None
    reviewer_id: int | None = None
    benchmark_question_id: int | None = None
    update_date: datetime | None = None


class ChatLogEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    creation_date: str
    user_id: int | None = None
    user_email: str | None = None
    conversation_id: int | None = None
    question: str
    answer: str
    sources: list[Any] | None = None
    latency_ms: int | None = None
    model_used: str | None = None
    breadth: str | None = None
    domain: str | None = None
    outcome: AnswerOutcome = AnswerOutcome.GENERATED
    error_code: str | None = None
    candidates_count: int | None = None
    top_rerank_score: float | None = None
    condensed_query: str | None = None
    sub_queries: list[str] | None = None
    refined_query: str | None = None
    aux_tokens: int | None = None
    retrieval_ms: int | None = None
    rerank_ms: int | None = None
    config_hash: str | None = None
    config_version: str | None = None
    evidence_profile: str | None = None
    answer_style: str | None = None
    finish_reason: str | None = None
    message_id: int | None = None
    space_id: int = 0
    pack_key: str | None = None
    depth_requested: str | None = None
    as_of_date: date | None = None
    confidence: float | None = None
    sources_count: int | None = None
    ttft_ms: int | None = None
    generation_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    feedback: FeedbackResponse | None = None
    review: ChatLogReviewResponse | None = None


class ChatLogReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ReviewStatus
    note: str | None = Field(None, max_length=MAX_REVIEW_NOTE_CHARS)


class HistoryTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=16000)


class BenchmarkDraftResponse(BaseModel):
    question: str
    history: list[HistoryTurn]
    as_of_date: date | None = None
    source_hint: str | None = None
    datasets: list[str]
    benchmark_question_id: int | None = None


class BenchmarkTransferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=16000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=100)
    expected_answer: str | None = Field(default=None, max_length=16000)
    source_hint: str | None = Field(default=None, max_length=5000)
    as_of_date: date | Literal[""] | None = None
    annotations: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list, max_length=100)
    notes: str | None = Field(default=None, max_length=2000)
    is_active: bool = False


class ChatLogsResponse(BaseModel):
    logs: list[ChatLogEntry]
    total: int
