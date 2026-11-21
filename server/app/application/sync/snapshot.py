"""What one full snapshot of a channel's storage publishes.

Shared by runs and by the adoption of ingested documents, so both apply exactly
the same folder, exclusions, ``.ragignore`` files, document types and size limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from domain.sync.ignore import (
    MAX_RAGIGNORE_BYTES,
    RAGIGNORE,
    IgnoreRules,
    InvalidIgnoreRulesError,
    relative_to,
)


if TYPE_CHECKING:
    from application.sync.ports import StorageEntry, StorageSnapshotSource
    from domain.sync.channel import SyncChannel


# Document names are stored in a 255-character column.
MAX_PATH_CHARS = 255


@dataclass
class SnapshotSelection:
    revision: str
    # Files the snapshot publishes, by path.
    desired: dict[str, StorageEntry] = field(default_factory=dict)
    # Excluded by the channel's patterns or a ``.ragignore``.
    excluded: list[str] = field(default_factory=list)
    # Too large or with a too long path: a published version stays until they fit again.
    unreadable: list[str] = field(default_factory=list)


async def select_snapshot(source: StorageSnapshotSource, channel: SyncChannel) -> SnapshotSelection:
    """Read the whole snapshot; raises ``StorageReadError`` or ``InvalidIgnoreRulesError``."""
    snapshot = await source.list_entries(channel, await source.resolve_revision(channel))
    # The revision actually read (for S3, a hash of the listing rather than the request).
    revision = snapshot.revision
    rules = IgnoreRules(
        channel.settings.exclusions, await read_ragignore(source, channel, revision, snapshot.entries)
    )
    selection = SnapshotSelection(revision)

    for entry in snapshot.entries:
        relative = relative_to(entry.path, channel.settings.folder)

        if not relative or not source.is_document(entry.path):
            continue

        if rules.is_excluded(relative):
            selection.excluded.append(entry.path)
        elif entry.size > source.max_file_bytes or len(entry.path) > MAX_PATH_CHARS:
            selection.unreadable.append(entry.path)
        else:
            selection.desired[entry.path] = entry

    return selection


async def read_ragignore(
    source: StorageSnapshotSource, channel: SyncChannel, revision: str, entries: tuple[StorageEntry, ...]
) -> dict[str, str]:
    """Every ``.ragignore`` under the channel folder, keyed by its folder relative to the channel.

    Any unreadable file fails the read: publishing without it would expose what it hides.
    """
    files: dict[str, str] = {}

    for entry in entries:
        relative = relative_to(entry.path, channel.settings.folder)

        if not relative or relative.rsplit("/", 1)[-1] != RAGIGNORE:
            continue

        if entry.size > MAX_RAGIGNORE_BYTES:
            raise InvalidIgnoreRulesError(entry.path, f"larger than {MAX_RAGIGNORE_BYTES} bytes")

        try:
            # The limit applies to the bytes actually read, not only to the listed size.
            content = await source.read(channel, revision, entry, MAX_RAGIGNORE_BYTES)
        except ValueError as exc:
            raise InvalidIgnoreRulesError(entry.path, f"larger than {MAX_RAGIGNORE_BYTES} bytes") from exc

        files[relative.rpartition("/")[0]] = content.decode("utf-8")

    return files
