"""Admin logs & chat logs schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


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
    sources: list | None = None
    latency_ms: int | None = None
    model_used: str | None = None
    breadth: str | None = None
    domain: str | None = None
    retrieval_count: int | None = None
    reranker_score: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatLogsResponse(BaseModel):
    logs: list[ChatLogEntry]
    total: int
