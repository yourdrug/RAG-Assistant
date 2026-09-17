"""API key schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ApiKeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100)


class ApiKeyCreateResponse(BaseModel):
    key: str
    key_prefix: str
    name: str | None


class ApiKeyRevokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str


class ApiKeyResponse(BaseModel):
    id: int
    key_prefix: str
    name: str | None
    creation_date: datetime | None
    last_used_at: datetime | None
    revoked_at: datetime | None
    is_active: bool
