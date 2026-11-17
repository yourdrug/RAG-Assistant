"""Admin logs & chat logs schemas."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict

from domain.rag.outcome import AnswerOutcome


class LogEntry(BaseModel):
    timestamp: str
    level: str
    logger: str
    request_id: str
    message: str
    filename: str | None = None
    lineno: int | None = None


class LogsResponse(BaseModel):
    logs: list[LogEntry]
    total: int


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


class ChatLogsResponse(BaseModel):
    logs: list[ChatLogEntry]
    total: int
