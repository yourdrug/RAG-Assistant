"""Group schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class GroupResponse(BaseModel):
    id: int
    name: str
    creation_date: datetime | None = None


class GroupMemberResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int = Field(validation_alias="user_id")
    email: str


class CreateGroupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100)


class GroupMemberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int
