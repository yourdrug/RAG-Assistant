"""Curator assignment schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CuratorAssignmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
