"""S3 prefix as the storage of a sync channel (the S3 implementation of the snapshot port).

A file's version is ``<size>_<LastModified>``, the identity CLI ingestion already
records, so documents adopted from a prefix keep their indexing. The revision is a
hash of the sorted listing. Objects have no external link.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from botocore.exceptions import BotoCoreError, ClientError

from application.ingestion.registry import s3_file_hash
from application.sync.ports import (
    SnapshotIncompleteError,
    StorageAccessLostError,
    StorageEntry,
    StorageSnapshot,
)


if TYPE_CHECKING:
    from application.ports.file_storage import FileItem
    from domain.sync.channel import SyncChannel

LATEST = "latest"
# Access problems: the run keeps every document instead of treating the prefix as empty.
ACCESS_ERROR_CODES = frozenset(
    {"AccessDenied", "NoSuchBucket", "InvalidAccessKeyId", "SignatureDoesNotMatch"}
)


@runtime_checkable
class ObjectListing(Protocol):
    async def list_all_files(self, prefix: str) -> list[FileItem]: ...

    async def read_file(self, key: str, max_bytes: int) -> bytes: ...


def prefix_of(channel: SyncChannel) -> str:
    folder = channel.settings.folder
    return f"{folder}/" if folder else ""


class S3SnapshotSource:
    def __init__(self, storage: ObjectListing, extensions: tuple[str, ...], max_file_bytes: int) -> None:
        self._storage = storage
        self._extensions = frozenset(extension.lower() for extension in extensions)
        self._max_file_bytes = max_file_bytes

    def is_document(self, path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        return "." in name and f".{name.rsplit('.', 1)[-1].lower()}" in self._extensions

    @property
    def max_file_bytes(self) -> int:
        return self._max_file_bytes

    async def resolve_revision(self, channel: SyncChannel) -> str:
        """A bucket has no commits: the snapshot's revision is the hash of its listing."""
        del channel
        return LATEST

    async def list_entries(self, channel: SyncChannel, revision: str) -> StorageSnapshot:
        del revision
        try:
            items = await self._storage.list_all_files(prefix_of(channel))
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")

            if code in ACCESS_ERROR_CODES:
                message = f"No access to the bucket ({code})"
                raise StorageAccessLostError(message) from exc

            message = f"Listing was cut short ({code or type(exc).__name__})"
            raise SnapshotIncompleteError(message) from exc
        except (BotoCoreError, OSError) as exc:
            message = f"Listing was cut short ({type(exc).__name__})"
            raise SnapshotIncompleteError(message) from exc

        entries = tuple(StorageEntry(item.key, s3_file_hash(item), item.size_bytes) for item in items)
        digest = hashlib.sha256("\n".join(f"{e.path}\0{e.version}" for e in entries).encode()).hexdigest()
        return StorageSnapshot(digest[:40], entries)

    async def read(self, channel: SyncChannel, revision: str, entry: StorageEntry) -> bytes:
        del channel, revision
        return await self._storage.read_file(entry.path, self._max_file_bytes)

    def source_link(self, channel: SyncChannel, revision: str, path: str) -> str | None:
        del channel, revision, path
        return None
