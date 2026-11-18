"""Knowledge space schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class KnowledgeSpaceResponse(BaseModel):
    id: int
    name: str
    pack_key: str
    can_upload: bool = Field(description="Whether the current user may fill the space with documents")
    upload_group_ids: list[int] | None = Field(
        None, description="Groups admitted to fill the space (ADMIN only)"
    )


class CreateKnowledgeSpaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100)
    pack_key: str = Field(
        ..., min_length=1, max_length=64, description="Loaded domain pack; fixed at creation"
    )
    upload_group_ids: list[int] = Field(default_factory=list)


class UpdateKnowledgeSpaceRequest(BaseModel):
    """Rename a space or change its admitted groups; the pack cannot change."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(None, min_length=1, max_length=100)
    upload_group_ids: list[int] | None = None


class DomainPacksResponse(BaseModel):
    pack_keys: list[str]


class AssignClientSpaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space_ids: list[PositiveInt] = Field(
        ..., max_length=100, description="Support spaces the client may ask in; empty clears them"
    )
