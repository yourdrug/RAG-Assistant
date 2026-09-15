"""Chat & conversation schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=16000)
    conversation_id: int | None = None
    depth: Literal["narrow", "broad"] | None = None
    as_of_date: date | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: int
    sources: list[dict] | None = None
    confidence: float | None = Field(None, description="Answer confidence score (0.0-1.0)")
    input_tokens: int | None = None
    output_tokens: int | None = None


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    sources: list | None
    creation_date: datetime | None


class NewConversationResponse(BaseModel):
    conversation_id: int


class ConversationHistoryResponse(BaseModel):
    conversation_id: int
    messages: list[MessageResponse]


class ConversationListItem(BaseModel):
    id: int
    title: str | None = None
    created_at: datetime | None = None
    message_count: int = 0


class ConversationListResponse(BaseModel):
    conversations: list[ConversationListItem]
