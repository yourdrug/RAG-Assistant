"""Pydantic models for request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    kind: str


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    kind: str
    is_active: bool


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str = "user"
    kind: str = "internal"


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: int
    sources: list[dict]


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class NewConversationResponse(BaseModel):
    conversation_id: int


class MessageResponse(BaseModel):
    role: str
    content: str


class ConversationHistoryResponse(BaseModel):
    conversation_id: int
    messages: list[MessageResponse]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    id: int
    filename: str
    visibility: str
    status: str
    error_message: str | None = None
    chunks: int | None = None
    chars: int | None = None


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    files: list[str]


class IngestStatusResponse(BaseModel):
    status: str
    mode: str | None = None
    file: str | None = None
    force: bool | None = None
    docs_dir: str | None = None


class IngestRegistryItem(BaseModel):
    filename: str
    chunks: int
    chars: int
    indexed_at: str
    source: str


class IngestRegistryResponse(BaseModel):
    total_files: int
    total_chunks: int
    files: list[IngestRegistryItem]


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


class CreateGroupRequest(BaseModel):
    name: str


class GroupResponse(BaseModel):
    id: int
    name: str


class GroupMemberResponse(BaseModel):
    id: int
    email: str


class GroupMemberRequest(BaseModel):
    user_id: int


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class AssignClientRequest(BaseModel):
    internal_user_id: int


class ClientAssignmentResponse(BaseModel):
    internal_user_id: int
    email: str
    assigned_at: datetime


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    api: str
    qdrant: str
    ollama: str
    ollama_models: list[str] | None = None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    questions_path: str | None = None
    out_dir: str | None = None
    top_k: int | None = None
    judge_model: str | None = None


class BenchmarkResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


class StatusResponse(BaseModel):
    status: str
    detail: str | None = None
    id: int | None = None
    is_active: bool | None = None
