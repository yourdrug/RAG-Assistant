"""Sync channel: one storage location kept in line with documents of one knowledge space.

The storage is the source of truth. Visibility, group and owner apply to the whole
channel; different access means different channels. Widening what a channel publishes
(another location or branch, a broader folder, a removed exclusion, another audience or
responsible user) is an administrator decision; the responsible user may only narrow it.
Every settings change starts a new generation, which fences writes of older runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from urllib.parse import quote

from domain.knowledge.enums import DocumentVisibility
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError
from domain.sync.ignore import IgnoreRules, normalize_folder, relative_to


class StorageKind(StrEnum):
    GITHUB = "github"
    S3 = "s3"


class ChannelDeletionMode(StrEnum):
    # Remove the channel's documents with it.
    DELETE_DOCUMENTS = "delete_documents"
    # Keep them as manual documents with their ids and access.
    UNLINK = "unlink"


@dataclass(frozen=True)
class GitHubLocation:
    """One branch of one repository on one GitHub instance (github.com or Enterprise Server).

    ``repository_id`` identifies the repository across renames; ``repository`` is
    the ``owner/name`` shown to people.
    """

    instance: str
    repository_id: int
    repository: str
    branch: str

    def __post_init__(self) -> None:
        if not self.instance.strip() or not self.branch.strip() or self.repository_id <= 0:
            msg = "GitHub location needs an instance, a repository id and a branch"
            raise ValidationError(msg)

    def file_url(self, revision: str, path: str) -> str:
        """The file as it was in ``revision``; stays valid after later changes and renames."""
        return f"https://{self.instance}/{self.repository}/blob/{quote(revision)}/{quote(path)}"

    @property
    def key(self) -> tuple[str, str, int, str]:
        """Channels with the same key read the same files."""
        return (StorageKind.GITHUB.value, self.instance.lower(), self.repository_id, self.branch)


@dataclass(frozen=True)
class S3Location:
    """The deployment's bucket; the channel folder is the key prefix inside it."""

    bucket: str

    def __post_init__(self) -> None:
        if not self.bucket.strip():
            msg = "S3 location needs a bucket"
            raise ValidationError(msg)

    def file_url(self, revision: str, path: str) -> str | None:
        """Objects have no public link: answers cite them by path only."""
        del revision, path
        return None

    @property
    def key(self) -> tuple[str, str]:
        return (StorageKind.S3.value, self.bucket)


type StorageLocation = GitHubLocation | S3Location


@dataclass(frozen=True)
class ChannelAudience:
    """Who may read the channel's documents; the same rules as an uploaded document."""

    visibility: DocumentVisibility
    owner_id: int | None = None
    group_id: int | None = None

    def __post_init__(self) -> None:
        owned = self.visibility in {
            DocumentVisibility.INTERNAL_PRIVATE,
            DocumentVisibility.CLIENT_PRIVATE,
            DocumentVisibility.CLIENT_PUBLIC,
        }

        if owned != (self.owner_id is not None):
            msg = f"visibility '{self.visibility}' {'requires' if owned else 'does not take'} an owner"
            raise ValidationError(msg)

        if (self.visibility == DocumentVisibility.INTERNAL_GROUP) != (self.group_id is not None):
            msg = "internal_group visibility requires a group, other visibilities do not take one"
            raise ValidationError(msg)


@dataclass(frozen=True)
class ChannelSettings:
    """Everything that decides which files a channel publishes and to whom."""

    location: StorageLocation
    audience: ChannelAudience
    # ``None`` once the user is deleted: runs are rejected until an administrator assigns one.
    responsible_user_id: int | None
    folder: str = ""
    exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "folder", normalize_folder(self.folder))
        object.__setattr__(
            self, "exclusions", tuple(line.strip() for line in self.exclusions if line.strip())
        )
        # Parse now: a channel never stores patterns it cannot apply.
        self.rules()

        if isinstance(self.location, S3Location) and not self.folder:
            # The bucket also holds the application's own objects (uploads, previews).
            msg = "An S3 channel needs a folder: it never reads the whole bucket"
            raise ValidationError(msg, field="folder")

    def rules(self, ragignore: dict[str, str] | None = None) -> IgnoreRules:
        return IgnoreRules(self.exclusions, ragignore)

    def widened_by(self, new: ChannelSettings) -> list[str]:
        """What ``new`` publishes beyond ``self``; empty when it only narrows or keeps the scope."""
        reasons = []

        if new.location != self.location:
            reasons.append("storage location or branch")

        if relative_to(new.folder, self.folder) is None:
            reasons.append("folder")

        added = set(new.exclusions) - set(self.exclusions)

        if set(self.exclusions) - set(new.exclusions) or any(line.startswith("!") for line in added):
            # A removed exclusion or a new re-inclusion lets more files in.
            reasons.append("exclusions")

        if new.audience != self.audience:
            reasons.append("audience")

        if new.responsible_user_id != self.responsible_user_id:
            reasons.append("responsible user")

        return reasons


@dataclass
class SyncChannel:
    space_id: int
    settings: ChannelSettings
    id: int | None = None
    # Fences writes: a run publishes only while the channel is still at its generation.
    generation: int = 1
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def storage_kind(self) -> StorageKind:
        return StorageKind.S3 if isinstance(self.settings.location, S3Location) else StorageKind.GITHUB

    def can_manage(self, user_id: int, *, is_admin: bool) -> bool:
        return is_admin or (
            self.settings.responsible_user_id is not None and user_id == self.settings.responsible_user_id
        )

    def change_settings(self, new: ChannelSettings, *, user_id: int, is_admin: bool) -> None:
        if not self.can_manage(user_id, is_admin=is_admin):
            msg = "Only an administrator or the channel's responsible user can change it"
            raise BusinessRuleViolationError(msg)

        widened = self.settings.widened_by(new)

        if widened and not is_admin:
            msg = f"Only an administrator can widen a channel ({', '.join(widened)})"
            raise BusinessRuleViolationError(msg)

        if new != self.settings:
            self.settings = new
            self.generation += 1

    def with_settings(self, settings: ChannelSettings) -> SyncChannel:
        """A copy for checks against other channels before a change is applied."""
        return replace(self, settings=settings)

    def covers(self, path: str) -> bool:
        """``path`` (relative to the storage root) is in this channel's folder and not excluded by it."""
        relative = relative_to(path, self.settings.folder)
        return (
            bool(relative)
            and relative is not None
            and not self.settings.rules().excludes_by_channel(relative)
        )
