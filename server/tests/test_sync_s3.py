"""S3 prefixes as sync channels: the snapshot port contract, runs, adoption of ingested documents."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
import pytest

from application.ingestion import DocumentSyncService
from application.ingestion.registry import s3_file_hash
from application.ports.file_storage import FileItem
from application.sync import SyncChannelService, SyncRunRequests
from application.sync.channels import (
    SKIP_MANUAL_EDITS,
    SKIP_NOT_PUBLISHED,
    UNLINKED_WITHOUT_SOURCE_WARNING,
)
from application.sync.ports import (
    EntryChangedError,
    SnapshotIncompleteError,
    StorageAccessLostError,
    StorageSources,
)
from application.sync.runs import SyncRunService
from domain.ingestion.raw_document import RawDocument
from domain.ingestion.repositories import IngestionRegistryEntry
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.shared.exceptions import BusinessRuleViolationError, ValidationError
from domain.sync.channel import (
    ChannelAudience,
    ChannelDeletionMode,
    ChannelSettings,
    S3Location,
    StorageKind,
    SyncChannel,
)
from domain.sync.run import FileOutcomeKind, SyncRunStatus, SyncTrigger
from infrastructure.storage.file_storage import ObjectChangedError, StoredObject
from infrastructure.storage.s3_snapshot_source import S3SnapshotSource
from tests.fakes import DEFAULT_SPACE_ID, FakeUnitOfWorkFactory, admit_all_users
from tests.test_sync_runs import ADMIN, RESPONSIBLE, FakeStorage, TextProcessor


if TYPE_CHECKING:
    from application.ingestion.dto import RegistryRecord
    from application.sync.ports import StorageReadError, StorageSnapshotSource
    from tests.fakes import FakeUnitOfWork


def etag_of(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()  # noqa: S324 - S3 ETags of single-part uploads


class Bucket:
    """In-memory objects of a bucket; a modification bumps LastModified."""

    def __init__(self, files: dict[str, str]) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.clock = 0
        self.error: Exception | None = None
        self.buckets_read: list[str] = []

        for key, text in files.items():
            self.put(key, text)

    def put(self, key: str, text: str) -> None:
        self.clock += 1
        self.objects[key] = (text.encode(), f"2026-11-21T10:00:{self.clock:02d}+00:00")

    def item(self, key: str) -> FileItem:
        content, modified = self.objects[key]
        return FileItem(key, key.rsplit("/", 1)[-1], len(content), modified, "")

    def stored(self, key: str) -> StoredObject:
        content, modified = self.objects[key]
        return StoredObject(key, len(content), modified, etag_of(content))

    async def list_objects(self, bucket: str, prefix: str) -> list[StoredObject]:
        self.buckets_read.append(bucket)

        if self.error is not None:
            raise self.error

        return [self.stored(key) for key in sorted(self.objects) if key.startswith(prefix)]

    async def copy_object_in(self, source_bucket: str, source_key: str, key: str, *, etag: str) -> None:
        self.buckets_read.append(source_bucket)

        if source_key not in self.objects or etag_of(self.objects[source_key][0]) != etag:
            raise ObjectChangedError(source_key)

        self.objects[key] = self.objects[source_key]

    async def read_object(self, bucket: str, key: str, max_bytes: int, *, etag: str) -> bytes:
        self.buckets_read.append(bucket)

        if key not in self.objects or etag_of(self.objects[key][0]) != etag:
            raise ObjectChangedError(key)

        content = self.objects[key][0]

        if len(content) > max_bytes:
            message = "too large"
            raise ValueError(message)

        return content


def s3_settings(folder: str = "docs", **changes: object) -> ChannelSettings:
    values: dict[str, object] = {
        "location": S3Location("rag-documents"),
        "audience": ChannelAudience(DocumentVisibility.INTERNAL_PUBLIC),
        "responsible_user_id": RESPONSIBLE.user_id,
        "folder": folder,
        **changes,
    }
    return ChannelSettings(**values)  # type: ignore[arg-type]


def s3_source(bucket: Bucket) -> S3SnapshotSource:
    return S3SnapshotSource(bucket, (".md", ".txt"), max_file_bytes=1000)


# --- port contract ----------------------------------------------------------------------------


async def check_snapshot_contract(
    source: StorageSnapshotSource, channel: SyncChannel, write: dict[str, str], change: tuple[str, str]
) -> None:
    """Every handler: files under the folder only, stable versions, readable content."""
    snapshot = await source.list_entries(channel, await source.resolve_revision(channel))
    paths = {entry.path for entry in snapshot.entries}
    assert paths == set(write)
    entry = next(e for e in snapshot.entries if e.path == change[0])
    assert await source.read(channel, snapshot.revision, entry) == write[change[0]].encode()
    again = await source.list_entries(channel, await source.resolve_revision(channel))
    assert again == snapshot


@pytest.mark.asyncio
async def test_github_fake_and_s3_handler_satisfy_the_snapshot_contract() -> None:
    files = {"docs/a.md": "Альфа", "docs/sub/b.md": "Бета"}
    github_channel = SyncChannel(DEFAULT_SPACE_ID, s3_settings(), id=1)
    await check_snapshot_contract(FakeStorage(files), github_channel, files, ("docs/a.md", "x"))

    bucket = Bucket({**files, "other/c.md": "Гамма"})
    await check_snapshot_contract(s3_source(bucket), github_channel, files, ("docs/a.md", "x"))


@pytest.mark.asyncio
async def test_s3_versions_and_revision_follow_object_changes() -> None:
    bucket = Bucket({"docs/a.md": "Альфа", "docs/.ragignore": "drafts/\n", "docs/img.png": "png"})
    source = s3_source(bucket)
    channel = SyncChannel(DEFAULT_SPACE_ID, s3_settings(), id=1)
    first = await source.list_entries(channel, "latest")
    # .ragignore and non-document files are listed: the run decides what they mean.
    assert {e.path for e in first.entries} == {"docs/a.md", "docs/.ragignore", "docs/img.png"}
    entry = next(e for e in first.entries if e.path == "docs/a.md")
    assert entry.version == etag_of("Альфа".encode())
    # The version CLI ingestion recorded travels along for adopted documents.
    assert entry.legacy_version == s3_file_hash(bucket.item("docs/a.md"))
    assert (
        source.is_document("docs/a.md"),
        source.is_document("docs/img.png"),
        source.is_document("docs/.ragignore"),
    ) == (
        True,
        False,
        False,
    )

    bucket.put("docs/a.md", "Альфа v2")
    second = await source.list_entries(channel, "latest")
    assert second.revision != first.revision
    assert source.source_link(channel, second.revision, "docs/a.md") is None


@pytest.mark.parametrize(
    "error,expected",
    [
        (ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2"), StorageAccessLostError),
        (ClientError({"Error": {"Code": "InternalError"}}, "ListObjectsV2"), SnapshotIncompleteError),
        (OSError("connection reset"), SnapshotIncompleteError),
    ],
)
@pytest.mark.asyncio
async def test_failed_listing_is_an_error_not_an_empty_prefix(
    error: Exception, expected: type[StorageReadError]
) -> None:
    bucket = Bucket({"docs/a.md": "Альфа"})
    bucket.error = error

    with pytest.raises(expected):
        await s3_source(bucket).list_entries(SyncChannel(DEFAULT_SPACE_ID, s3_settings(), id=1), "latest")


# --- runs and adoption --------------------------------------------------------------------


class S3World:
    def __init__(self, files: dict[str, str]) -> None:
        self.factory = FakeUnitOfWorkFactory()
        self.uow: FakeUnitOfWork = self.factory._uow
        admit_all_users(self.uow)

        for user in (ADMIN, RESPONSIBLE):
            self.uow.fake_users.add_user(user.user_id)

        self.bucket = Bucket(files)
        self.channels = SyncChannelService(
            self.factory, sources=StorageSources({StorageKind.S3: s3_source(self.bucket)})
        )
        self.requests = SyncRunRequests(self.factory)
        self.runs = SyncRunService(
            self.factory,
            StorageSources({StorageKind.S3: s3_source(self.bucket)}),
            lambda _key: TextProcessor(self.factory),
        )
        self.channel_id = 0

    async def create(self, **changes: Any) -> None:
        channel = await self.channels.create(DEFAULT_SPACE_ID, s3_settings(**changes), ADMIN)
        self.channel_id = channel.id or 0

    async def sync(self) -> SyncRunStatus | None:
        run = await self.requests.request(self.channel_id, SyncTrigger.MANUAL, ADMIN)
        assert run is not None
        return await self.runs.execute(run.id or 0)

    async def documents(self) -> dict[str, Document]:
        documents = await self.uow.documents.list_channel_documents(self.channel_id)
        return {doc.sync.path: doc for doc in documents if doc.sync is not None}

    async def last_outcomes(self) -> dict[str, FileOutcomeKind]:
        [run, *_] = await self.uow.sync_runs.list_for_channel(self.channel_id)
        return {outcome.path: outcome.kind for outcome in run.outcomes}

    async def ingested(
        self,
        key: str,
        *,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        owner_id: int | None = None,
    ) -> Document:
        document = await self.uow.documents.save(
            Document(
                filename=key.rsplit("/", 1)[-1],
                ingestion_key=key,
                visibility=visibility,
                owner_id=owner_id,
                status=DocumentStatus.DONE,
            )
        )

        if key in self.bucket.objects:
            await self.uow.ingestion_registry.upsert(
                IngestionRegistryEntry(
                    filename=key,
                    file_hash=s3_file_hash(self.bucket.item(key)),
                    source=f"s3://b/{key}",
                    chunks=1,
                    chars=1,
                    ingestion_key=key,
                )
            )

        return document


@pytest.mark.asyncio
async def test_s3_channel_publishes_updates_and_deletes_after_a_full_listing() -> None:
    world = S3World(
        {"docs/a.md": "Альфа", "docs/b.txt": "Бета", "docs/img.png": "png", "other/c.md": "Гамма"}
    )
    await world.create()
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert sorted(await world.documents()) == ["docs/a.md", "docs/b.txt"]

    world.bucket.put("docs/a.md", "Альфа v2")
    del world.bucket.objects["docs/b.txt"]
    world.bucket.error = OSError("connection reset")
    assert await world.sync() == SyncRunStatus.FAILED
    assert sorted(await world.documents()) == ["docs/a.md", "docs/b.txt"]

    world.bucket.error = None
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert await world.last_outcomes() == {
        "docs/a.md": FileOutcomeKind.UPDATED,
        "docs/b.txt": FileOutcomeKind.DELETED,
    }


@pytest.mark.asyncio
async def test_adoption_keeps_ids_access_and_indexing_and_is_idempotent() -> None:
    world = S3World({"docs/a.md": "Альфа", "docs/b.md": "Бета", "docs/stale.md": "Старое"})
    adopted = await world.ingested("docs/a.md")
    private = await world.ingested(
        "docs/b.md", visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=RESPONSIBLE.user_id
    )
    stale = await world.ingested("docs/stale.md")
    outside = await world.ingested("other/c.md")
    world.bucket.put("docs/stale.md", "Изменено после загрузки")
    await world.create()

    with pytest.raises(BusinessRuleViolationError):
        await world.channels.adopt_ingested(world.channel_id, RESPONSIBLE)

    report = await world.channels.adopt_ingested(world.channel_id, ADMIN)
    assert (report.adopted, report.different_access) == (["docs/a.md", "docs/stale.md"], ["docs/b.md"])
    documents = await world.documents()
    assert (documents["docs/a.md"].id, documents["docs/a.md"].visibility) == (adopted.id, adopted.visibility)
    assert (await world.uow.documents.get_by_id(private.id or 0)).sync is None  # type: ignore[union-attr]
    assert (await world.uow.documents.get_by_id(outside.id or 0)).sync is None  # type: ignore[union-attr]

    again = await world.channels.adopt_ingested(world.channel_id, ADMIN)
    assert again.adopted == []

    # The private assignment stays manual, so the channel publishes its own document for b.md.
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    outcomes = await world.last_outcomes()
    assert outcomes["docs/a.md"] == FileOutcomeKind.UNCHANGED
    assert outcomes["docs/stale.md"] == FileOutcomeKind.UPDATED
    assert (await world.documents())["docs/stale.md"].id == stale.id


def record(key: str) -> RegistryRecord:
    return {"source": f"s3://b/{key}", "hash": "h", "chunks": 0, "chars": 0}


@pytest.mark.asyncio
async def test_cli_ingestion_refuses_keys_of_an_s3_channel() -> None:
    world = S3World({"docs/a.md": "Альфа"})
    await world.create(exclusions=("drafts/",))
    sync = DocumentSyncService(world.factory)

    with pytest.raises(BusinessRuleViolationError, match="sync channel"):
        await sync.ensure_document_rows({"docs/new.md": record("docs/new.md")})

    # Keys the channel excludes or leaves outside its prefix are still ingested as before.
    ids = await sync.ensure_document_rows(
        {"docs/drafts/x.md": record("docs/drafts/x.md"), "other/y.md": record("other/y.md")}
    )
    assert sorted(ids) == ["docs/drafts/x.md", "other/y.md"]


@pytest.mark.asyncio
async def test_github_and_s3_channels_never_overlap_each_other() -> None:
    world = S3World({})
    await world.create(folder="docs")

    with pytest.raises(BusinessRuleViolationError, match="overlap"):
        await world.channels.create(DEFAULT_SPACE_ID, s3_settings("docs/api"), ADMIN)

    await world.channels.create(DEFAULT_SPACE_ID, s3_settings("guides"), ADMIN)


# --- scope, bucket and version pinning ------------------------------------------------------


@pytest.mark.asyncio
async def test_s3_channel_never_reads_the_application_storage_area() -> None:
    world = S3World({"uploads/users/7/1_secret.md": "Секрет", "docs/a.md": "Альфа"})

    with pytest.raises(ValidationError):
        await world.create(folder="")

    for folder in ("uploads", "uploads/users", "previews/x"):
        with pytest.raises(BusinessRuleViolationError, match="storage area"):
            await world.create(folder=folder)

    await world.create(folder="docs")

    with pytest.raises(BusinessRuleViolationError, match="storage area"):
        await world.channels.change(world.channel_id, s3_settings("uploads"), ADMIN)

    # A channel saved before the rule existed reads nothing at all.
    legacy = SyncChannel(DEFAULT_SPACE_ID, s3_settings("uploads/users"), id=9)

    with pytest.raises(StorageAccessLostError):
        await s3_source(world.bucket).list_entries(legacy, "latest")


@pytest.mark.asyncio
async def test_s3_channel_reads_the_bucket_it_was_bound_to() -> None:
    world = S3World({"docs/a.md": "Альфа"})
    await world.create(location=S3Location("archive-bucket"))
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert set(world.bucket.buckets_read) == {"archive-bucket"}


@pytest.mark.asyncio
async def test_different_objects_with_equal_size_and_timestamp_are_not_a_rename() -> None:
    world = S3World({"docs/a.md": "Альфа"})
    await world.create()
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    old = (await world.documents())["docs/a.md"]
    _, modified = world.bucket.objects.pop("docs/a.md")
    # Same size and LastModified, other content.
    world.bucket.objects["docs/b.md"] = ("Бетаа".encode()[: len("Альфа".encode())], modified)

    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert await world.last_outcomes() == {
        "docs/b.md": FileOutcomeKind.ADDED,
        "docs/a.md": FileOutcomeKind.DELETED,
    }
    assert (await world.documents())["docs/b.md"].id != old.id


@pytest.mark.asyncio
async def test_object_changed_after_listing_fails_the_file_and_is_retried() -> None:
    world = S3World({"docs/a.md": "Альфа"})
    await world.create()
    source = s3_source(world.bucket)
    channel = SyncChannel(DEFAULT_SPACE_ID, s3_settings(), id=world.channel_id)
    snapshot = await source.list_entries(channel, "latest")
    world.bucket.put("docs/a.md", "Альфа v2")

    with pytest.raises(EntryChangedError):
        await source.read(channel, snapshot.revision, snapshot.entries[0])

    # A .ragignore read is limited by its own size limit, whatever the listing said.
    with pytest.raises(ValueError, match="larger"):
        await source.read(channel, snapshot.revision, snapshot.entries[0], max_bytes=3)


@pytest.mark.asyncio
async def test_adoption_uses_the_runs_selection_and_never_hands_over_what_a_run_deletes() -> None:
    world = S3World(
        {
            "docs/a.md": "Альфа",
            "docs/drafts/x.md": "Черновик",
            "docs/.ragignore": "drafts/\n",
            "docs/b.md": "Б",
        }
    )
    kept = await world.ingested("docs/drafts/x.md")
    edited = await world.ingested("docs/b.md")
    await world.uow.documents.set_has_manual_edits(edited.id or 0, value=True)
    await world.ingested("docs/a.md")
    await world.create()

    report = await world.channels.adopt_ingested(world.channel_id, ADMIN)
    assert report.adopted == ["docs/a.md"]
    assert {item.path: item.reason for item in report.skipped} == {
        "docs/drafts/x.md": SKIP_NOT_PUBLISHED,
        "docs/b.md": SKIP_MANUAL_EDITS,
    }

    assert await world.sync() == SyncRunStatus.SUCCEEDED
    # The excluded document stayed manual and still exists.
    document = await world.uow.documents.get_by_id(kept.id or 0)
    assert document is not None
    assert document.sync is None


@pytest.mark.asyncio
async def test_adoption_moves_nothing_when_the_prefix_cannot_be_read() -> None:
    world = S3World({"docs/a.md": "Альфа"})
    document = await world.ingested("docs/a.md")
    await world.create()
    world.bucket.error = OSError("connection reset")

    with pytest.raises(SnapshotIncompleteError):
        await world.channels.adopt_ingested(world.channel_id, ADMIN)

    assert (await world.uow.documents.get_by_id(document.id or 0)).sync is None  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_unlinked_documents_keep_their_source_a_free_name_and_leave_cli_management() -> None:
    world = S3World({"docs/a.md": "Альфа", "docs/b.md": "Бета"})
    adopted = await world.ingested("docs/b.md")
    await world.create()
    await world.channels.adopt_ingested(world.channel_id, ADMIN)
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    documents = await world.documents()
    # A manual document took the channel document's name and audience meanwhile.
    manual = await world.uow.documents.save(
        Document(filename="docs/a.md", space_id=DEFAULT_SPACE_ID, status=DocumentStatus.DONE)
    )

    assert await world.channels.delete(world.channel_id, ChannelDeletionMode.UNLINK, ADMIN) == 2
    unlinked = await world.uow.documents.get_by_id(documents["docs/a.md"].id or 0)
    assert unlinked is not None
    assert (unlinked.sync, unlinked.filename) == (None, "docs/a(1).md")
    assert unlinked.source_path.startswith("uploads/public/")
    assert world.bucket.objects[unlinked.source_path][0] == "Альфа".encode()
    assert (await world.uow.documents.get_by_id(manual.id or 0)).filename == "docs/a.md"  # type: ignore[union-attr]

    # The adopted CLI document is no longer CLI-managed: its ingestion key is gone.
    cli = await world.uow.documents.get_by_id(adopted.id or 0)
    assert cli is not None
    assert (cli.sync, cli.ingestion_key) == (None, None)


@pytest.mark.asyncio
async def test_unlinked_document_without_a_kept_source_says_it_cannot_be_reindexed() -> None:
    world = S3World({"docs/a.md": "Альфа"})
    await world.create()
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    [document] = (await world.documents()).values()
    world.bucket.put("docs/a.md", "Изменено")

    await world.channels.delete(world.channel_id, ChannelDeletionMode.UNLINK, ADMIN)
    unlinked = await world.uow.documents.get_by_id(document.id or 0)
    assert unlinked is not None
    assert (unlinked.source_path, unlinked.warning_message) == ("", UNLINKED_WITHOUT_SOURCE_WARNING)


@pytest.mark.asyncio
async def test_cli_indexing_refuses_a_document_adopted_after_its_rows_were_resolved() -> None:
    world = S3World({"docs/a.md": "Альфа"})
    document = await world.ingested("docs/a.md")
    sync = DocumentSyncService(world.factory)
    await world.create()
    await world.channels.adopt_ingested(world.channel_id, ADMIN)

    with pytest.raises(BusinessRuleViolationError, match="sync channel"):
        await sync.sync_one_document(
            "a.md",
            document.id or 0,
            "s3://b/docs/a.md",
            [RawDocument("CLI", {})],
            pack_document=None,
            visibility=DocumentVisibility.INTERNAL_PUBLIC,
            owner_id=None,
            group_id=None,
        )
