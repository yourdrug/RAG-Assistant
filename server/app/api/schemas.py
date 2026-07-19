"""
api/schemas.py — Pydantic models for request/response schemas.
"""

from pydantic import BaseModel


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


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: int
    sources: list


class NewConversationResponse(BaseModel):
    conversation_id: int


class UploadResponse(BaseModel):
    files: list[str]


class DocumentResponse(BaseModel):
    id: int
    filename: str
    visibility: str
    status: str
    error_message: str | None = None
    chunks: int | None = None
    chars: int | None = None


class CreateGroupRequest(BaseModel):
    name: str


class GroupMemberRequest(BaseModel):
    user_id: int


class AssignClientRequest(BaseModel):
    internal_user_id: int
