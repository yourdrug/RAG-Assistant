"""Storage handlers (Strategy): everything that depends on the kind of storage.

A run knows only this port. A handler resolves the channel's branch to a revision,
lists the files of that revision under the channel folder, reads their content and
links to a file in that revision. Adding a storage kind means a new implementation of
the port and a registry entry; reconciliation does not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from domain.shared.exceptions import AppServerError


if TYPE_CHECKING:
    from collections.abc import Mapping

    from domain.sync.channel import StorageKind, SyncChannel


class StorageReadError(AppServerError):
    """The storage could not be read as a whole; a run changes nothing after it."""


class RevisionNotFoundError(StorageReadError):
    """The branch does not exist (anymore)."""


class StorageAccessLostError(StorageReadError):
    """The handler has no access to the repository or bucket (anymore)."""


class SnapshotIncompleteError(StorageReadError):
    """The listing was cut short; files missing from it must not be deleted."""


class StorageHandlerMissingError(StorageReadError):
    """No handler is configured for the channel's storage kind."""


class EntryChangedError(StorageReadError):
    """A file changed after the listing was read; its listed version cannot be read anymore."""


@dataclass(frozen=True)
class StorageEntry:
    """One file of a revision; ``path`` is relative to the storage root."""

    path: str
    # Identifies the content: equal versions mean equal bytes (a blob sha for GitHub,
    # the ETag for S3). Never a weak signature such as size and modification time.
    version: str
    size: int
    # The version in an earlier, weaker format (S3: size and LastModified, as CLI
    # ingestion records it). A document published under it is moved to ``version``
    # once instead of being re-indexed; it never makes two paths the same content.
    legacy_version: str | None = None


@dataclass(frozen=True)
class StorageSnapshot:
    """Every file of one revision under the channel folder, read completely."""

    revision: str
    entries: tuple[StorageEntry, ...]


class StorageSnapshotSource(Protocol):
    def is_document(self, path: str) -> bool:
        """Files of this storage kind that become documents (``*.md`` for GitHub)."""
        ...

    @property
    def max_file_bytes(self) -> int: ...

    async def resolve_revision(self, channel: SyncChannel) -> str:
        """The current revision of the channel's branch; raises ``StorageReadError``."""
        ...

    async def list_entries(self, channel: SyncChannel, revision: str) -> StorageSnapshot:
        """All files under the channel folder; raises instead of returning a partial listing."""
        ...

    async def read(
        self, channel: SyncChannel, revision: str, entry: StorageEntry, max_bytes: int | None = None
    ) -> bytes:
        """Exactly the listed version of the file, at most ``max_bytes`` (default: ``max_file_bytes``).

        Raises ``EntryChangedError`` when that version cannot be read anymore.
        """
        ...

    def source_link(self, channel: SyncChannel, revision: str, path: str) -> str | None: ...


class StorageSources:
    """Handlers by storage kind; composed in Composition."""

    def __init__(self, sources: Mapping[StorageKind, StorageSnapshotSource]) -> None:
        self._sources = dict(sources)

    def for_channel(self, channel: SyncChannel) -> StorageSnapshotSource:
        source = self._sources.get(channel.storage_kind)

        if source is None:
            msg = f"No storage handler is configured for {channel.storage_kind.value} channels"
            raise StorageHandlerMissingError(msg)

        return source
