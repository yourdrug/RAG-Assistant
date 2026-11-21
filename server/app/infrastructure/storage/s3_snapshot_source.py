"""S3 prefix as the storage of a sync channel (the S3 implementation of the snapshot port).

A file's version is its ETag: it changes with the content, and two different objects
do not share one, so a rename is never inferred from coincidental size and timestamp.
CLI ingestion recorded ``<size>_<LastModified>``; that legacy version is reported
next to it so adopted documents keep their indexing. Reads are pinned to the listed
ETag and to the bucket the channel was bound to. The revision is a hash of the sorted
listing. Objects have no external link.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from botocore.exceptions import BotoCoreError, ClientError

from application.sync.ports import (
    EntryChangedError,
    SnapshotIncompleteError,
    StorageAccessLostError,
    StorageEntry,
    StorageSnapshot,
)
from application.sync.scope import reserved_overlap
from domain.sync.channel import S3Location
from infrastructure.storage.file_storage import ObjectChangedError


if TYPE_CHECKING:
    from domain.sync.channel import SyncChannel
    from infrastructure.storage.file_storage import StoredObject

LATEST = "latest"
# Access problems: the run keeps every document instead of treating the prefix as empty.
ACCESS_ERROR_CODES = frozenset(
    {"AccessDenied", "NoSuchBucket", "InvalidAccessKeyId", "SignatureDoesNotMatch"}
)


@runtime_checkable
class ObjectListing(Protocol):
    async def list_objects(self, bucket: str, prefix: str) -> list[StoredObject]: ...

    async def read_object(self, bucket: str, key: str, max_bytes: int, *, etag: str) -> bytes: ...

    async def copy_object_in(self, source_bucket: str, source_key: str, key: str, *, etag: str) -> None: ...


def prefix_of(channel: SyncChannel) -> str:
    folder = channel.settings.folder
    return f"{folder}/" if folder else ""


def legacy_version(item: StoredObject) -> str:
    """The version CLI ingestion records (``s3_file_hash``)."""
    return f"{item.size}_{item.last_modified}"


def bucket_of(channel: SyncChannel) -> str:
    location = channel.settings.location

    if not isinstance(location, S3Location):
        message = "Not an S3 channel"
        raise StorageAccessLostError(message)

    return location.bucket


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
        reserved = reserved_overlap(channel.settings.folder)

        if reserved is not None:
            # Saved before the rule existed: read nothing rather than the application's objects.
            message = f"The channel folder includes the application's storage area '{reserved}'"
            raise StorageAccessLostError(message)

        try:
            items = await self._storage.list_objects(bucket_of(channel), prefix_of(channel))
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

        if any(not item.etag for item in items):
            message = "Listing has objects without an ETag"
            raise SnapshotIncompleteError(message)

        entries = tuple(
            StorageEntry(item.key, item.etag, item.size, legacy_version=legacy_version(item))
            for item in items
        )
        digest = hashlib.sha256("\n".join(f"{e.path}\0{e.version}" for e in entries).encode()).hexdigest()
        return StorageSnapshot(digest[:40], entries)

    async def read(
        self, channel: SyncChannel, revision: str, entry: StorageEntry, max_bytes: int | None = None
    ) -> bytes:
        del revision
        limit = self._max_file_bytes if max_bytes is None else max_bytes

        if entry.size > limit:
            message = f"Object {entry.path} is larger than {limit} bytes"
            raise ValueError(message)

        try:
            content = await self._storage.read_object(
                bucket_of(channel), entry.path, limit, etag=entry.version
            )
        except ObjectChangedError as exc:
            raise EntryChangedError(str(exc)) from exc

        if len(content) != entry.size:
            message = f"Object {entry.path} changed after it was listed"
            raise EntryChangedError(message)

        return content

    async def preserve(self, channel: SyncChannel, path: str, version: str, key: str) -> bool:
        """Copy the version ``version`` of ``path`` to the managed storage key ``key``."""
        try:
            await self._storage.copy_object_in(bucket_of(channel), path, key, etag=version)
        except ObjectChangedError:
            return False

        return True

    def source_link(self, channel: SyncChannel, revision: str, path: str) -> str | None:
        del channel, revision, path
        return None
