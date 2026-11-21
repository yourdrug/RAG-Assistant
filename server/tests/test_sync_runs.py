"""Sync runs: reconciliation of a channel with full storage snapshots."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.documents.dto import PersistDocumentCommand, PreparedDocument
from application.documents.persistence import persist_document_result
from application.rag.dto import CompletedAnswer, SourceMetadata, StorageSource
from application.rag.service import RagService
from application.sync import SyncChannelService, SyncRunRequests
from application.sync.ports import SnapshotIncompleteError, StorageEntry, StorageSnapshot, StorageSources
from application.sync.runs import SyncRunService
from application.sync.sources import SyncSourceLinks
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.enums import DocumentStatus, DocumentVisibility
from domain.knowledge.index_sync import OutboxOperation
from domain.runtime.jobs import JobType
from domain.shared.exceptions import BusinessRuleViolationError
from domain.sync.channel import ChannelAudience, ChannelSettings, GitHubLocation, StorageKind
from domain.sync.reconcile import PublishedFile, reconcile
from domain.sync.run import FileOutcomeKind, SyncRunStatus, SyncTrigger
from tests.fakes import DEFAULT_SPACE_ID, FakeUnitOfWorkFactory, admit_all_users


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from application.documents.dto import PersistGuard
    from domain.knowledge.document import Document
    from domain.sync.channel import SyncChannel
    from tests.fakes import FakeUnitOfWork


ADMIN = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
RESPONSIBLE = UserContext(user_id=2, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
OTHER = UserContext(user_id=3, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)


class FakeStorage:
    """A repository branch: files by path; the version is the content hash."""

    is_document = staticmethod(lambda path: path.endswith(".md"))
    max_file_bytes = 1000

    def __init__(self, files: dict[str, str]) -> None:
        self.files = {path: text.encode() for path, text in files.items()}
        self.revision = "c1"
        self.cut_short = False
        self.on_read: Callable[[StorageEntry], Awaitable[None]] | None = None
        self.reads: list[str] = []

    async def resolve_revision(self, channel: SyncChannel) -> str:
        del channel
        return self.revision

    async def list_entries(self, channel: SyncChannel, revision: str) -> StorageSnapshot:
        del channel

        if self.cut_short:
            message = "tree truncated"
            raise SnapshotIncompleteError(message)

        return StorageSnapshot(
            revision,
            tuple(
                StorageEntry(path, hashlib.sha1(content).hexdigest(), len(content))  # noqa: S324 - a blob id
                for path, content in sorted(self.files.items())
            ),
        )

    async def read(self, channel: SyncChannel, revision: str, entry: StorageEntry) -> bytes:
        del channel, revision
        self.reads.append(entry.path)

        if self.on_read is not None:
            await self.on_read(entry)

        return self.files[entry.path]

    def source_link(self, channel: SyncChannel, revision: str, path: str) -> str | None:
        return channel.settings.location.file_url(revision, path)


class TextProcessor:
    """One chunk per file, persisted through the real persist transaction and its guard."""

    def __init__(self, factory: FakeUnitOfWorkFactory) -> None:
        self.factory = factory
        self.broken: set[str] = set()

    async def prepare(
        self, path: Path, filename: str, document_id: int, doc_domain: str | None
    ) -> PreparedDocument:
        del document_id, doc_domain

        if filename in self.broken:
            message = "parser failed"
            raise ValueError(message)

        text = await asyncio.to_thread(path.read_text)
        return PreparedDocument(
            raw_chunks=[RawDocument(text, {"source": filename})],
            doc_domain="general",
            pack_document=None,
            warnings=(),
            quality=None,
        )

    async def persist_prepared(
        self, document: Document, prepared: PreparedDocument, *, guard: PersistGuard | None = None
    ) -> bool:
        persisted = await persist_document_result(
            self.factory,
            PersistDocumentCommand(
                document_id=document.id or 0,
                original_filename=document.filename,
                raw_chunks=prepared.raw_chunks,
                visibility=document.visibility,
                owner_id=document.owner_id,
                group_id=document.group_id,
                doc_domain=prepared.doc_domain,
                replace_id=None,
                warning_message=None,
                quality=None,
                storage_deletes=[],
                guard=guard,
            ),
            domain_registry=None,
            pack_hooks=None,
        )
        return persisted is not None


class World:
    def __init__(self, files: dict[str, str], exclusions: tuple[str, ...] = ()) -> None:
        self.factory = FakeUnitOfWorkFactory()
        self.uow: FakeUnitOfWork = self.factory._uow
        admit_all_users(self.uow)

        for user in (ADMIN, RESPONSIBLE, OTHER):
            self.uow.fake_users.add_user(user.user_id)

        self.storage = FakeStorage(files)
        self.processor = TextProcessor(self.factory)
        self.invalidator = AsyncMock()
        self.channels = SyncChannelService(self.factory, cache_invalidator=self.invalidator)
        self.runs = SyncRunService(
            self.factory, StorageSources({StorageKind.GITHUB: self.storage}), lambda _key: self.processor
        )
        self.requests = SyncRunRequests(self.factory)
        self.exclusions = exclusions
        self.channel_id = 0

    def settings(self, **changes: object) -> ChannelSettings:
        values: dict[str, object] = {
            "location": GitHubLocation("github.com", 42, "acme/docs", "main"),
            "audience": ChannelAudience(DocumentVisibility.INTERNAL_PUBLIC),
            "responsible_user_id": RESPONSIBLE.user_id,
            "folder": "docs",
            "exclusions": self.exclusions,
            **changes,
        }
        return ChannelSettings(**values)  # type: ignore[arg-type]

    async def create(self) -> None:
        channel = await self.channels.create(DEFAULT_SPACE_ID, self.settings(), ADMIN)
        self.channel_id = channel.id or 0

    async def sync(self) -> SyncRunStatus | None:
        run = await self.requests.request(self.channel_id, SyncTrigger.MANUAL, RESPONSIBLE)
        assert run is not None
        assert run.id is not None
        return await self.runs.execute(run.id)

    async def documents(self) -> dict[str, Document]:
        documents = await self.uow.documents.list_channel_documents(self.channel_id)
        return {doc.sync.path: doc for doc in documents if doc.sync is not None}

    async def texts(self) -> dict[str, list[str]]:
        result = {}

        for path, doc in (await self.documents()).items():
            rows, _ = await self.uow.chunks.list_for_document(doc.id or 0)
            staged = await self.uow.chunks.get_staged_revision(doc.id or 0)

            if staged is not None:
                await self.uow.chunks.publish_staged_revision(doc.id or 0, staged.token)
                rows, _ = await self.uow.chunks.list_for_document(doc.id or 0)

            result[path] = [row.content for row in rows]

        return result

    async def last_outcomes(self) -> dict[str, FileOutcomeKind]:
        [run, *_] = await self.uow.sync_runs.list_for_channel(self.channel_id)
        return {outcome.path: outcome.kind for outcome in run.outcomes}


async def synced_world(files: dict[str, str], exclusions: tuple[str, ...] = ()) -> World:
    world = World(files, exclusions)
    await world.create()
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    return world


# --- reconciliation rule ------------------------------------------------------------------


def test_only_an_unambiguous_move_is_a_rename() -> None:
    published = {
        "a.md": PublishedFile(1, "v1"),
        "b.md": PublishedFile(2, "v2"),
        "c.md": PublishedFile(3, "v2"),
    }
    plan = reconcile({"x.md": "v1", "y.md": "v2", "a2.md": "v9"}, published)
    assert [(r.document_id, r.old_path, r.new_path) for r in plan.renamed] == [(1, "a.md", "x.md")]
    # Two files with the same content disappeared: the new one is not a rename of either.
    assert plan.added == ["a2.md", "y.md"]
    assert plan.deleted == {"b.md": 2, "c.md": 3}


# --- runs -----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_run_publishes_documents_of_the_folder_only() -> None:
    world = await synced_world(
        {"docs/a.md": "Альфа", "docs/sub/b.md": "Бета", "docs/img.png": "png", "other/c.md": "Гамма"},
        exclusions=("drafts/",),
    )
    documents = await world.documents()
    assert sorted(documents) == ["docs/a.md", "docs/sub/b.md"]
    assert {path: doc.sync.storage_revision for path, doc in documents.items() if doc.sync} == {
        "docs/a.md": "c1",
        "docs/sub/b.md": "c1",
    }
    assert documents["docs/a.md"].visibility == DocumentVisibility.INTERNAL_PUBLIC
    assert await world.texts() == {"docs/a.md": ["Альфа"], "docs/sub/b.md": ["Бета"]}
    jobs = [job for job in world.uow.fake_background_jobs.jobs.values() if job.job_type == JobType.SYNC_RUN]
    assert len(jobs) == 1

    # Unchanged storage: nothing is read or re-indexed again.
    world.storage.reads.clear()
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert world.storage.reads == []
    assert set((await world.last_outcomes()).values()) == {FileOutcomeKind.UNCHANGED}


@pytest.mark.asyncio
async def test_cancelled_run_publishes_and_deletes_nothing() -> None:
    world = await synced_world({"docs/a.md": "Альфа", "docs/b.md": "Бета"})
    world.storage.files = {"docs/a.md": "Альфа v2".encode()}
    run_a = await world.requests.request(world.channel_id, SyncTrigger.MANUAL, RESPONSIBLE)
    assert run_a is not None

    async def newer_run_starts(entry: StorageEntry) -> None:
        del entry
        world.storage.on_read = None
        await world.requests.request(world.channel_id, SyncTrigger.WEBHOOK)

    world.storage.on_read = newer_run_starts
    assert await world.runs.execute(run_a.id or 0) == SyncRunStatus.CANCELLED
    assert await world.texts() == {"docs/a.md": ["Альфа"], "docs/b.md": ["Бета"]}

    [run_b] = await world.uow.sync_runs.list_unfinished(world.channel_id)
    assert await world.runs.execute(run_b.id or 0) == SyncRunStatus.SUCCEEDED
    assert await world.texts() == {"docs/a.md": ["Альфа v2"]}


@pytest.mark.asyncio
async def test_settings_change_fences_a_queued_run() -> None:
    world = await synced_world({"docs/a.md": "Альфа"})
    run = await world.requests.request(world.channel_id, SyncTrigger.MANUAL, RESPONSIBLE)
    assert run is not None
    await world.channels.change(world.channel_id, world.settings(folder="docs/api"), RESPONSIBLE)
    assert await world.runs.execute(run.id or 0) is None
    stored = await world.uow.sync_runs.get_by_id(run.id or 0)
    assert stored is not None
    assert stored.status == SyncRunStatus.CANCELLED
    assert sorted(await world.documents()) == ["docs/a.md"]


@pytest.mark.asyncio
async def test_failed_file_is_retried_by_the_next_run_at_the_same_commit() -> None:
    world = World({"docs/a.md": "Альфа", "docs/b.md": "Бета"})
    await world.create()
    world.processor.broken.add("docs/b.md")
    assert await world.sync() == SyncRunStatus.PARTIAL
    documents = await world.documents()
    assert documents["docs/b.md"].status == DocumentStatus.FAILED
    assert documents["docs/b.md"].sync is not None
    assert documents["docs/b.md"].sync.storage_version is None
    assert (await world.last_outcomes())["docs/b.md"] == FileOutcomeKind.FAILED

    world.processor.broken.clear()
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert (await world.last_outcomes())["docs/b.md"] == FileOutcomeKind.UPDATED
    assert await world.texts() == {"docs/a.md": ["Альфа"], "docs/b.md": ["Бета"]}


@pytest.mark.asyncio
async def test_unambiguous_rename_keeps_the_document() -> None:
    world = await synced_world({"docs/a.md": "Альфа"})
    before = (await world.documents())["docs/a.md"]
    world.storage.files = {"docs/renamed.md": "Альфа".encode()}
    world.storage.revision = "c2"
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    after = (await world.documents())["docs/renamed.md"]
    assert (after.id, after.filename) == (before.id, "docs/renamed.md")
    assert after.sync is not None
    assert after.sync.storage_revision == "c2"
    assert (await world.last_outcomes()) == {"docs/renamed.md": FileOutcomeKind.RENAMED}
    assert any(
        entry.operation == OutboxOperation.RENAME_DOCUMENT and entry.aggregate_id == before.id
        for entry in world.uow.fake_vector_outbox._entries.values()
    )


@pytest.mark.asyncio
async def test_empty_folder_in_a_full_snapshot_deletes_but_a_cut_short_one_does_not() -> None:
    world = await synced_world({"docs/a.md": "Альфа", "docs/b.md": "Бета"})
    world.storage.files = {}
    world.storage.cut_short = True
    assert await world.sync() == SyncRunStatus.FAILED
    assert sorted(await world.documents()) == ["docs/a.md", "docs/b.md"]

    world.storage.cut_short = False
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert await world.documents() == {}
    assert set((await world.last_outcomes()).values()) == {FileOutcomeKind.DELETED}


@pytest.mark.asyncio
async def test_ragignore_lines_remove_and_restore_files() -> None:
    world = await synced_world({"docs/a.md": "Альфа", "docs/b.md": "Бета"})
    world.storage.files["docs/.ragignore"] = b"b.md\n"
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert sorted(await world.documents()) == ["docs/a.md"]
    # Excluded now, so its published document is deleted.
    assert (await world.last_outcomes())["docs/b.md"] == FileOutcomeKind.DELETED

    world.storage.files["docs/.ragignore"] = b"# nothing hidden\n"
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert sorted(await world.documents()) == ["docs/a.md", "docs/b.md"]


@pytest.mark.parametrize("content", [b"!\n", b"\xff\xfe broken"], ids=["invalid-pattern", "not-utf8"])
@pytest.mark.asyncio
async def test_unreadable_ragignore_changes_nothing(content: bytes) -> None:
    world = await synced_world({"docs/a.md": "Альфа", "docs/b.md": "Бета"})
    world.storage.files = {"docs/a.md": "Альфа v2".encode(), "docs/.ragignore": content}
    assert await world.sync() == SyncRunStatus.FAILED
    assert await world.texts() == {"docs/a.md": ["Альфа"], "docs/b.md": ["Бета"]}


@pytest.mark.asyncio
async def test_run_is_rejected_when_the_responsible_user_lost_access() -> None:
    world = await synced_world({"docs/a.md": "Альфа"})
    world.storage.files = {}
    world.uow.fake_users._users[RESPONSIBLE.user_id].is_active = False
    run = await world.requests.request(world.channel_id, SyncTrigger.SCHEDULE)
    assert run is not None
    assert await world.runs.execute(run.id or 0) is None
    stored = await world.uow.sync_runs.get_by_id(run.id or 0)
    assert stored is not None
    assert (stored.status, stored.error) == (
        SyncRunStatus.REJECTED,
        "the responsible user is not an active internal user",
    )
    assert sorted(await world.documents()) == ["docs/a.md"]


@pytest.mark.asyncio
async def test_only_the_responsible_user_or_an_administrator_starts_a_run() -> None:
    world = await synced_world({"docs/a.md": "Альфа"})

    with pytest.raises(BusinessRuleViolationError):
        await world.requests.request(world.channel_id, SyncTrigger.MANUAL, OTHER)

    assert await world.requests.request(world.channel_id, SyncTrigger.MANUAL, ADMIN) is not None


@pytest.mark.asyncio
async def test_redelivered_webhook_does_not_restart_the_run() -> None:
    world = await synced_world({"docs/a.md": "Альфа"})
    first = await world.requests.request(world.channel_id, SyncTrigger.WEBHOOK, delivery_id="d-1")
    assert first is not None
    assert await world.requests.request(world.channel_id, SyncTrigger.WEBHOOK, delivery_id="d-1") is None
    [unfinished] = await world.uow.sync_runs.list_unfinished(world.channel_id)
    assert unfinished.id == first.id


@pytest.mark.asyncio
async def test_narrowed_audience_applies_to_documents_at_once() -> None:
    world = await synced_world({"docs/a.md": "Альфа"})
    group = ChannelAudience(DocumentVisibility.INTERNAL_GROUP, group_id=world.uow.fake_groups.member_of[0])
    await world.channels.change(world.channel_id, world.settings(audience=group), ADMIN)
    document = (await world.documents())["docs/a.md"]
    assert (document.visibility, document.group_id) == (DocumentVisibility.INTERNAL_GROUP, group.group_id)
    rows, _ = await world.uow.chunks.list_for_document(document.id or 0)
    assert {(row.visibility, row.group_id) for row in rows} == {
        (DocumentVisibility.INTERNAL_GROUP.value, group.group_id)
    }
    world.invalidator.invalidate_by_document_ids.assert_awaited_once_with([document.id], raise_on_error=True)

    # The next run re-indexes it so the vector payloads carry the new audience too.
    assert await world.sync() == SyncRunStatus.SUCCEEDED
    assert (await world.last_outcomes()) == {"docs/a.md": FileOutcomeKind.UPDATED}


@pytest.mark.asyncio
async def test_scheduled_reconciliation_queues_only_idle_channels() -> None:
    world = await synced_world({"docs/a.md": "Альфа"})
    assert await world.requests.request_scheduled() == 1
    assert await world.requests.request_scheduled() == 0


# --- answer sources -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_links_open_the_cited_revision_even_after_a_rename() -> None:
    world = await synced_world({"docs/a b.md": "Альфа"})
    document = (await world.documents())["docs/a b.md"]
    links = SyncSourceLinks(world.factory)
    cited = (await links.describe([document.id or 0, 999]))[document.id or 0]
    assert (cited.branch, cited.path, cited.revision) == ("main", "docs/a b.md", "c1")
    assert cited.url == "https://github.com/acme/docs/blob/c1/docs/a%20b.md"

    world.storage.files = {"docs/renamed.md": "Альфа".encode()}
    world.storage.revision = "c2"
    await world.sync()
    current = (await links.describe([document.id or 0]))[document.id or 0]
    assert current.url == "https://github.com/acme/docs/blob/c2/docs/renamed.md"
    # An answer stored before the rename keeps its link, which still opens the cited file.
    assert cited.url.endswith("/blob/c1/docs/a%20b.md")


@pytest.mark.asyncio
async def test_answer_sources_get_links_only_for_the_documents_they_already_cite() -> None:
    describe = AsyncMock(return_value={7: StorageSource("main", "docs/a.md", "c1", "https://github.com/x")})
    service = RagService(
        **dict.fromkeys(
            (
                "preparation",
                "safety",
                "cache",
                "retrieval",
                "evidence",
                "context",
                "generation",
                "finalization",
                "corpus",
            ),
            MagicMock(),
        ),
        storage_sources=MagicMock(describe=describe),
    )
    completed = CompletedAnswer(
        answer=MagicMock(),
        sources=(
            SourceMetadata(source="a.md", document_id=7),
            SourceMetadata(source="manual.pdf", document_id=8),
        ),
        confidence=0.9,
    )
    linked = await service.link_storage_sources(completed)
    describe.assert_awaited_once_with([7, 8])
    assert linked.sources[0]["storage_url"] == "https://github.com/x"
    assert "storage_url" not in linked.sources[1]
