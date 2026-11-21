"""Request and execute sync runs: reconcile a channel with a full snapshot of its storage.

One run per channel: a new request cancels the unfinished ones. A run reads the storage
without any open transaction, then applies each change in its own short transaction.
Every write is fenced: it locks the channel row and proceeds only while the channel is
still at the run's generation and the run is still running, so a settings change, a
newer run or a deleted channel stops an older run before its next write. Deletion
happens only after the whole snapshot was read; a cut-short listing changes nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Protocol

from application.documents.delete import delete_document_in_uow
from application.sync.audience import apply_audience, audience_matches
from application.sync.ports import StorageReadError
from application.sync.rights import channel_rights_problem
from domain.knowledge.document import Document, SyncBinding
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.index_sync import OutboxOperation, VectorOutboxEntry
from domain.shared.exceptions import AppError, BusinessRuleViolationError
from domain.sync.ignore import (
    MAX_RAGIGNORE_BYTES,
    RAGIGNORE,
    IgnoreRules,
    InvalidIgnoreRulesError,
    relative_to,
)
from domain.sync.reconcile import PublishedFile, reconcile
from domain.sync.run import FileOutcome, FileOutcomeKind, SyncRunStatus


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.documents.dto import PersistGuard, PreparedDocument
    from application.indexing.pack_hooks import DocumentHooksResolver
    from application.sync.ports import StorageEntry, StorageSnapshotSource, StorageSources
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.sync.channel import SyncChannel
    from domain.sync.reconcile import Rename
    from domain.sync.run import SyncRun


log = logging.getLogger("default")

# Document names are stored in a 255-character column.
MAX_PATH_CHARS = 255
MAX_DETAIL_CHARS = 300


class RunSupersededError(BusinessRuleViolationError):
    """The channel changed or a newer run started; this run must not write anymore."""

    def __init__(self) -> None:
        super().__init__("Sync run is no longer current")


class ContentProcessor(Protocol):
    """Parsing, splitting and persisting with the rules of one space's pack."""

    async def prepare(
        self, path: Path, filename: str, document_id: int, doc_domain: str | None
    ) -> PreparedDocument: ...

    async def persist_prepared(
        self, document: Document, prepared: PreparedDocument, *, guard: PersistGuard | None = None
    ) -> bool: ...


@dataclass
class RunState:
    """What one execution learned and did; outcomes are stored on the run at the end."""

    run: SyncRun
    channel: SyncChannel
    pack_key: str
    outcomes: list[FileOutcome] = field(default_factory=list)

    def record(self, path: str, kind: FileOutcomeKind, detail: str | None = None) -> None:
        self.outcomes.append(FileOutcome(path, kind, detail[:MAX_DETAIL_CHARS] if detail else None))


class SyncRunService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        sources: StorageSources,
        processors: Callable[[str], ContentProcessor],
        pack_hooks: DocumentHooksResolver | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._sources = sources
        self._processors = processors
        self._pack_hooks = pack_hooks

    # --- executing a run ------------------------------------------------------------------

    async def execute(self, run_id: int) -> SyncRunStatus | None:
        state = await self.begin(run_id)

        if state is None:
            return None

        try:
            await self.reconcile_channel(state)
        except RunSupersededError:
            log.info("Sync run %d superseded; stopping", run_id)
            await self.finish(state, SyncRunStatus.CANCELLED)
            return SyncRunStatus.CANCELLED
        except (StorageReadError, InvalidIgnoreRulesError, UnicodeDecodeError) as exc:
            log.warning("Sync run %d read no complete snapshot: %s", run_id, type(exc).__name__)
            await self.finish(state, SyncRunStatus.FAILED, error=describe(exc))
            return SyncRunStatus.FAILED
        except Exception as exc:
            # Never leave the run "running"; the job records the error itself.
            await self.finish(state, SyncRunStatus.FAILED, error=describe(exc))
            raise

        return await self.finish(state, None)

    async def begin(self, run_id: int) -> RunState | None:
        async with self._uow_factory.create(master=True) as uow:
            run = await uow.sync_runs.get_by_id(run_id)

            if run is None or run.status != SyncRunStatus.QUEUED:
                return None

            channel = await uow.sync_channels.get_by_id(run.channel_id, for_update=True)

            if channel is None or channel.generation != run.generation:
                run.cancel()
                await uow.sync_runs.update(run)
                return None

            space = await uow.knowledge_spaces.get_by_id(channel.space_id)
            problem = (
                await channel_rights_problem(uow, space, channel.settings)
                if space is not None
                else "space is gone"
            )

            if problem is not None:
                # Documents stay as they are until the rights are restored.
                run.reject(problem)
                await uow.sync_runs.update(run)
                log.warning("Sync run %d rejected: %s", run_id, problem)
                return None

            run.start()
            await uow.sync_runs.update(run)
            return RunState(run=run, channel=channel, pack_key=space.pack_key if space else "")

    async def finish(
        self, state: RunState, status: SyncRunStatus | None, error: str | None = None
    ) -> SyncRunStatus:
        async with self._uow_factory.create(master=True) as uow:
            run = await uow.sync_runs.get_by_id(state.run.id or 0)

            if run is None or run.is_finished:
                return run.status if run is not None else SyncRunStatus.CANCELLED

            run.storage_revision = state.run.storage_revision

            if status == SyncRunStatus.CANCELLED:
                run.outcomes = tuple(state.outcomes)
                run.cancel()
            elif status == SyncRunStatus.FAILED:
                run.fail(error or "storage could not be read")
            else:
                run.complete(tuple(state.outcomes))

            await uow.sync_runs.update(run)
            return run.status

    async def reconcile_channel(self, state: RunState) -> None:
        channel = state.channel
        source = self._sources.for_channel(channel)
        revision = await source.resolve_revision(channel)
        state.run.storage_revision = revision
        snapshot = await source.list_entries(channel, revision)
        rules = IgnoreRules(
            channel.settings.exclusions,
            await self.read_ragignore(source, channel, revision, snapshot.entries),
        )
        desired, keep = select_files(state, source, rules, snapshot.entries)
        documents = await self.channel_documents(channel)
        published = {
            path: PublishedFile(doc.id or 0, doc.sync.storage_version if doc.sync else None)
            for path, doc in documents.items()
            if path not in keep
        }
        stale_audience = {path for path, doc in documents.items() if not audience_matches(channel, doc)}
        plan = reconcile({path: entry.version for path, entry in desired.items()}, published)

        for rename in plan.renamed:
            await self.apply_rename(state, rename, desired[rename.new_path], revision)

        for path in [*plan.added, *plan.updated, *(p for p in plan.unchanged if p in stale_audience)]:
            await self.publish_file(state, source, desired[path], revision, documents.get(path))

        for path in plan.unchanged:
            if path not in stale_audience:
                state.record(path, FileOutcomeKind.UNCHANGED)

        for path, document_id in plan.deleted.items():
            await self.apply_delete(state, path, document_id)

    async def read_ragignore(
        self,
        source: StorageSnapshotSource,
        channel: SyncChannel,
        revision: str,
        entries: tuple[StorageEntry, ...],
    ) -> dict[str, str]:
        """Every ``.ragignore`` under the channel folder, keyed by its folder relative to the channel.

        Any unreadable file fails the run: publishing without it would expose what it hides.
        """
        files: dict[str, str] = {}

        for entry in entries:
            relative = relative_to(entry.path, channel.settings.folder)

            if not relative or relative.rsplit("/", 1)[-1] != RAGIGNORE:
                continue

            if entry.size > MAX_RAGIGNORE_BYTES:
                raise InvalidIgnoreRulesError(entry.path, f"larger than {MAX_RAGIGNORE_BYTES} bytes")

            content = await source.read(channel, revision, entry)
            files[relative.rpartition("/")[0]] = content.decode("utf-8")

        return files

    async def channel_documents(self, channel: SyncChannel) -> dict[str, Document]:
        async with self._uow_factory.create() as uow:
            documents = await uow.documents.list_channel_documents(channel.id or 0)

        return {doc.sync.path: doc for doc in documents if doc.sync is not None}

    # --- fenced writes --------------------------------------------------------------------

    async def fence(self, uow: UnitOfWork, state: RunState) -> None:
        """Lock the channel row; refuse to write once the run is no longer current."""
        channel = await uow.sync_channels.get_by_id(state.channel.id or 0, for_update=True)
        run = await uow.sync_runs.get_by_id(state.run.id or 0)

        if (
            channel is None
            or channel.generation != state.run.generation
            or run is None
            or run.status != SyncRunStatus.RUNNING
        ):
            raise RunSupersededError

    async def publish_file(
        self,
        state: RunState,
        source: StorageSnapshotSource,
        entry: StorageEntry,
        revision: str,
        existing: Document | None,
    ) -> None:
        """Index one file as a whole; until it is indexed, search keeps its previous version."""
        channel = state.channel
        try:
            document = existing or await self.create_document(state, entry.path)
            content = await source.read(channel, revision, entry)
            processor = self._processors(state.pack_key)
            prepared = await with_temp_file(
                content,
                entry.path,
                lambda path: processor.prepare(path, entry.path, document.id or 0, None),
            )
            binding = SyncBinding(channel.id or 0, entry.path, entry.version, revision)

            async def guard(uow: UnitOfWork) -> None:
                await self.fence(uow, state)
                await apply_audience(uow, channel, document)
                await uow.documents.set_sync_binding(document.id or 0, binding)

            audience = channel.settings.audience
            target = Document(
                id=document.id,
                filename=entry.path,
                visibility=audience.visibility,
                owner_id=audience.owner_id,
                group_id=audience.group_id,
                space_id=channel.space_id,
            )

            if not await processor.persist_prepared(target, prepared, guard=guard):
                return
        except RunSupersededError:
            raise
        except Exception as exc:  # noqa: BLE001 - One file's failure never cancels the others; recorded, logged.
            await self.mark_failed(state, existing, entry.path, exc)
            return

        state.record(entry.path, FileOutcomeKind.UPDATED if existing else FileOutcomeKind.ADDED)

    async def create_document(self, state: RunState, path: str) -> Document:
        channel = state.channel
        audience = channel.settings.audience
        async with self._uow_factory.create(master=True) as uow:
            await self.fence(uow, state)
            return await uow.documents.save(
                Document(
                    filename=path,
                    visibility=audience.visibility,
                    owner_id=audience.owner_id,
                    group_id=audience.group_id,
                    space_id=channel.space_id,
                    status=DocumentStatus.PROCESSING,
                    # No version yet: the next run retries the file until it is published.
                    sync=SyncBinding(channel.id or 0, path),
                )
            )

    async def mark_failed(
        self, state: RunState, existing: Document | None, path: str, exc: Exception
    ) -> None:
        log.warning("Sync run %s: file failed (%s)", state.run.id, type(exc).__name__)
        state.record(path, FileOutcomeKind.FAILED, describe(exc))

        if existing is not None:
            # A published version keeps serving search; only the failure is shown.
            return

        async with self._uow_factory.create(master=True) as uow:
            await self.fence(uow, state)
            documents = await uow.documents.list_channel_documents(state.channel.id or 0)
            created = next((d for d in documents if d.sync is not None and d.sync.path == path), None)

            if created is not None and created.id is not None:
                await uow.documents.update_status(
                    created.id, status=DocumentStatus.FAILED.value, error=describe(exc)
                )

    async def apply_rename(self, state: RunState, rename: Rename, entry: StorageEntry, revision: str) -> None:
        """Same content under a new path: keep the document, its id and its vectors."""
        async with self._uow_factory.create(master=True) as uow:
            await self.fence(uow, state)
            document = await uow.documents.get_by_id(rename.document_id, for_update=True)

            if document is None:
                return

            await uow.documents.update_filename(rename.document_id, rename.new_path, document.source_path)
            await uow.chunks.update_filename_by_document_id(rename.document_id, rename.new_path)
            await uow.documents.set_sync_binding(
                rename.document_id,
                SyncBinding(state.channel.id or 0, rename.new_path, entry.version, revision),
            )
            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.RENAME_DOCUMENT,
                    aggregate_type="document",
                    aggregate_id=rename.document_id,
                    payload={"document_id": rename.document_id},
                )
            )

        state.record(rename.new_path, FileOutcomeKind.RENAMED, f"from {rename.old_path}")

    async def apply_delete(self, state: RunState, path: str, document_id: int) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await self.fence(uow, state)
            document = await uow.documents.get_by_id(document_id, for_update=True)

            if document is None or document.sync is None or document.sync.channel_id != state.channel.id:
                return

            hooks = (
                await self._pack_hooks.for_document(uow, document) if self._pack_hooks is not None else None
            )
            await delete_document_in_uow(uow, document_id, hooks)

        state.record(path, FileOutcomeKind.DELETED)


def select_files(
    state: RunState, source: StorageSnapshotSource, rules: IgnoreRules, entries: tuple[StorageEntry, ...]
) -> tuple[dict[str, StorageEntry], set[str]]:
    """Files the snapshot publishes, and published paths kept although they cannot be read now."""
    desired: dict[str, StorageEntry] = {}
    keep: set[str] = set()

    for entry in entries:
        relative = relative_to(entry.path, state.channel.settings.folder)

        if not relative or not source.is_document(entry.path):
            continue

        if rules.is_excluded(relative):
            state.record(entry.path, FileOutcomeKind.EXCLUDED)
        elif entry.size > source.max_file_bytes or len(entry.path) > MAX_PATH_CHARS:
            # A published version stays until the file fits again.
            keep.add(entry.path)
            state.record(entry.path, FileOutcomeKind.FAILED, "file is too large or its path too long")
        else:
            desired[entry.path] = entry

    return desired, keep


async def with_temp_file[T](content: bytes, path: str, use: Callable[[Path], Awaitable[T]]) -> T:
    """Parsers read files: hand them the content in a temporary file named like the source."""
    suffix = Path(path).suffix

    def write() -> Path:
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as handle:
            handle.write(content)
            return Path(handle.name)

    temp = await asyncio.to_thread(write)
    try:
        return await use(temp)
    finally:
        await asyncio.to_thread(temp.unlink, missing_ok=True)


def describe(exc: BaseException) -> str:
    message = exc.message if isinstance(exc, AppError) else str(exc)
    return f"{type(exc).__name__}: {message}"[:MAX_DETAIL_CHARS]
