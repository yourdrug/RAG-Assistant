"""Request and response schemas of sync channels, their runs and the GitHub webhook."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from domain.knowledge.enums import DocumentVisibility
from domain.sync.channel import ChannelDeletionMode, StorageKind
from domain.sync.run import FileOutcomeKind, SyncRunStatus, SyncTrigger


class ChannelSettingsRequest(BaseModel):
    """Everything a channel publishes and to whom.

    The GitHub instance and the S3 bucket are the deployment's; an S3 channel's folder
    is the key prefix inside the bucket.
    """

    model_config = ConfigDict(extra="forbid")

    storage_kind: StorageKind = StorageKind.GITHUB
    repository_id: PositiveInt | None = None
    repository: str | None = Field(default=None, min_length=3, max_length=200, pattern=r"^[\w.-]+/[\w.-]+$")
    branch: str | None = Field(default=None, min_length=1, max_length=255)
    folder: str = Field(default="", max_length=1000)
    exclusions: list[str] = Field(default_factory=list, max_length=200)
    visibility: DocumentVisibility
    owner_id: PositiveInt | None = None
    group_id: PositiveInt | None = None
    responsible_user_id: PositiveInt

    @model_validator(mode="after")
    def location_matches_kind(self) -> ChannelSettingsRequest:
        github_fields = (self.repository_id, self.repository, self.branch)

        if self.storage_kind == StorageKind.GITHUB and None in github_fields:
            msg = "A GitHub channel needs repository_id, repository and branch"
            raise ValueError(msg)

        if self.storage_kind == StorageKind.S3 and any(value is not None for value in github_fields):
            msg = "An S3 channel takes no repository fields"
            raise ValueError(msg)

        return self


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
    storage_kind: StorageKind
    instance: str | None = None
    repository_id: int | None = None
    repository: str | None = None
    branch: str | None = None
    bucket: str | None = None
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


class AdoptionResponse(BaseModel):
    adopted: list[str]
    different_access: list[str]
    skipped: list[str]


class WebhookResponse(BaseModel):
    queued: int


class StorageKindsResponse(BaseModel):
    """Storage kinds this deployment can synchronize; others cannot be created."""

    kinds: list[StorageKind]
