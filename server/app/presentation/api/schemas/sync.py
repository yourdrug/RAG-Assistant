"""Request and response schemas of sync channels, their runs and the GitHub webhook."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from domain.knowledge.enums import DocumentVisibility
from domain.sync.channel import ChannelDeletionMode
from domain.sync.run import FileOutcomeKind, SyncRunStatus, SyncTrigger


class ChannelSettingsRequest(BaseModel):
    """Everything a channel publishes and to whom; the GitHub instance is the deployment's."""

    model_config = ConfigDict(extra="forbid")

    repository_id: PositiveInt
    repository: str = Field(min_length=3, max_length=200, pattern=r"^[\w.-]+/[\w.-]+$")
    branch: str = Field(min_length=1, max_length=255)
    folder: str = Field(default="", max_length=1000)
    exclusions: list[str] = Field(default_factory=list, max_length=200)
    visibility: DocumentVisibility
    owner_id: PositiveInt | None = None
    group_id: PositiveInt | None = None
    responsible_user_id: PositiveInt


class CreateChannelRequest(ChannelSettingsRequest):
    space_id: PositiveInt


class FileOutcomeResponse(BaseModel):
    path: str
    kind: FileOutcomeKind
    detail: str | None = None


class SyncRunResponse(BaseModel):
    id: int
    status: SyncRunStatus
    trigger: SyncTrigger
    storage_revision: str | None
    error: str | None
    counts: dict[str, int]
    # Failed files and conflicts; the full list of unchanged files is not repeated.
    problems: list[FileOutcomeResponse]
    creation_date: datetime
    started_at: datetime | None
    finished_at: datetime | None


class SyncChannelResponse(BaseModel):
    id: int
    space_id: int
    storage_kind: str
    instance: str
    repository_id: int
    repository: str
    branch: str
    folder: str
    exclusions: list[str]
    visibility: DocumentVisibility
    owner_id: int | None
    group_id: int | None
    responsible_user_id: int | None
    generation: int
    last_run: SyncRunResponse | None = None


class DeleteChannelResponse(BaseModel):
    mode: ChannelDeletionMode
    documents: int


class WebhookResponse(BaseModel):
    queued: int
