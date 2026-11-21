"""Create, change and delete sync channels.

Only an administrator binds a storage location or widens a channel; its responsible
user may narrow it. Channel changes of one space are serialized on the space row, so
two concurrent changes can never create overlapping channels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from application.documents.delete import delete_document_in_uow
from application.documents.utils import generate_storage_key
from application.sync.audience import apply_audience, audience_matches
from application.sync.ports import StorageHandlerMissingError
from application.sync.rights import channel_rights_problem
from application.sync.scope import ensure_channel_scope
from application.sync.snapshot import select_snapshot
from domain.knowledge.document import SyncBinding
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.index_sync import OutboxOperation, VectorOutboxEntry
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError
from domain.sync.channel import ChannelDeletionMode, S3Location, StorageKind, SyncChannel
from domain.sync.ownership import ensure_no_overlap


if TYPE_CHECKING:
    from application.indexing.pack_hooks import DocumentHooksResolver
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.sync.ports import StorageSources
    from application.sync.snapshot import SnapshotSelection
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.identity.user_context import UserContext
    from domain.knowledge.document import Document
    from domain.knowledge.space import KnowledgeSpace
    from domain.sync.channel import ChannelSettings
    from domain.sync.run import SyncRun


log = logging.getLogger("default")


@dataclass(frozen=True)
class SkippedPath:
    path: str
    reason: str


@dataclass
class AdoptionReport:
    adopted: list[str] = field(default_factory=list)
    # Access differs from the channel's: the document stays manual.
    different_access: list[str] = field(default_factory=list)
    # Left as they are, with the reason (the channel would not publish the file,
    # another document holds the path, the document has manual chunk edits).
    skipped: list[SkippedPath] = field(default_factory=list)


SKIP_NOT_PUBLISHED = "the channel does not publish this file (excluded, unsupported or missing)"
SKIP_UNREADABLE = "the file is too large or its path too long"
SKIP_BOUND = "another document of the channel holds this path"
SKIP_MANUAL_EDITS = "the document has manual chunk edits; re-indexing would discard them"


@dataclass(frozen=True)
class ChannelView:
    channel: SyncChannel
    last_run: SyncRun | None


class SyncChannelService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        pack_hooks: DocumentHooksResolver | None = None,
        cache_invalidator: CacheInvalidatorPort | None = None,
        storage_kinds: frozenset[StorageKind] = frozenset(StorageKind),
        sources: StorageSources | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._sources = sources
        self._pack_hooks = pack_hooks
        self._cache_invalidator = cache_invalidator
        # Kinds with a configured storage handler: a channel of another kind could never run.
        self._storage_kinds = storage_kinds

    @property
    def storage_kinds(self) -> frozenset[StorageKind]:
        return self._storage_kinds

    def ensure_runnable(self, channel: SyncChannel) -> None:
        if channel.storage_kind not in self._storage_kinds:
            msg = f"{channel.storage_kind.value} channels are not available in this deployment"
            raise BusinessRuleViolationError(msg)

    async def create(self, space_id: int, settings: ChannelSettings, actor: UserContext) -> SyncChannel:
        if not actor.is_admin:
            msg = "Only an administrator can create a sync channel"
            raise BusinessRuleViolationError(msg)

        channel = SyncChannel(space_id=space_id, settings=settings)
        self.ensure_runnable(channel)

        async with self._uow_factory.create(master=True) as uow:
            space = await lock_space(uow, space_id)
            await self.validate(uow, space, channel)
            channel = await uow.sync_channels.add(channel)

        log.info("Sync channel %s created in space %d by user %d", channel.id, space_id, actor.user_id)
        return channel

    async def change(self, channel_id: int, settings: ChannelSettings, actor: UserContext) -> SyncChannel:
        """Apply new settings; a non-administrator may only narrow the channel."""
        async with self._uow_factory.create(master=True) as uow:
            channel = await existing_channel(uow, channel_id)
            space = await lock_space(uow, channel.space_id)
            channel = await existing_channel(uow, channel_id, for_update=True)
            previous = channel.settings

            if isinstance(previous.location, S3Location) and isinstance(settings.location, S3Location):
                # A channel stays bound to its bucket, whatever bucket is configured now.
                settings = replace(settings, location=previous.location)

            channel.change_settings(settings, user_id=actor.user_id, is_admin=actor.is_admin)
            self.ensure_runnable(channel)

            changed_acl: list[int] = []

            if channel.settings != previous:
                await self.validate(uow, space, channel)
                await uow.sync_channels.update(channel)

                if channel.settings.audience != previous.audience:
                    changed_acl = await self.apply_audience(uow, channel, previous)

        if changed_acl and self._cache_invalidator is not None:
            # Answers cached for the previous audience must not be served anymore.
            await self._cache_invalidator.invalidate_by_document_ids(changed_acl, raise_on_error=True)

        return channel

    @staticmethod
    async def apply_audience(uow: UnitOfWork, channel: SyncChannel, previous: ChannelSettings) -> list[int]:
        """The new audience applies at once to documents the new scope still covers.

        A document the changed scope may no longer cover keeps the audience approved
        for it until the next run re-publishes it from the new snapshot or deletes it:
        a widened audience must never reach files outside the scope it was granted for.
        The next run rewrites the vectors.
        """
        changed = []
        same_location = previous.location == channel.settings.location

        for document in await uow.documents.list_channel_documents(channel.id or 0):
            if not same_location or document.sync is None or not channel.covers(document.sync.path):
                continue

            if await apply_audience(uow, channel, document) and document.id is not None:
                changed.append(document.id)
                # Forget the published version: the next run re-indexes the file, so the
                # vector payloads carry the new audience too (unchanged vectors are reused).
                await uow.documents.set_storage_version(document.id, None, None)
                # A revision being indexed carries the previous audience in its payloads.
                await discard_staged_revision(uow, document.id)

        return changed

    async def delete(self, channel_id: int, mode: ChannelDeletionMode, actor: UserContext) -> int:
        """Delete the channel with its documents, or unlink them into manual documents.

        Unlinked documents keep their ids and access. Their published source file is
        copied to managed storage first (no transaction is open meanwhile), so a later
        re-index can read it; a document whose source could not be kept says so. A name
        taken by a manual document meanwhile gets a numbered suffix.
        Returns how many documents were deleted or unlinked.
        """
        if not actor.is_admin:
            msg = "Only an administrator can delete a sync channel"
            raise BusinessRuleViolationError(msg)

        preserved = await self.preserve_sources(channel_id) if mode == ChannelDeletionMode.UNLINK else {}

        async with self._uow_factory.create(master=True) as uow:
            channel = await existing_channel(uow, channel_id)
            await lock_space(uow, channel.space_id)
            await existing_channel(uow, channel_id, for_update=True)
            documents = await uow.documents.list_channel_documents(channel_id)

            for document in documents:
                if document.id is None:
                    continue

                if mode == ChannelDeletionMode.UNLINK:
                    await unlink_document(uow, document, preserved.get(document.id))
                else:
                    hooks = (
                        await self._pack_hooks.for_document(uow, document)
                        if self._pack_hooks is not None
                        else None
                    )
                    await delete_document_in_uow(uow, document.id, hooks)

            await uow.sync_channels.delete(channel_id)

        log.info("Sync channel %d deleted (%s, %d documents)", channel_id, mode.value, len(documents))
        return len(documents)

    async def preserve_sources(self, channel_id: int) -> dict[int, str]:
        """Copy each published source file to the key a manual upload of it would have."""
        async with self._uow_factory.create(master=True) as uow:
            channel = await existing_channel(uow, channel_id)
            documents = await uow.documents.list_channel_documents(channel_id)

        if self._sources is None:
            return {}

        try:
            source = self._sources.for_channel(channel)
        except StorageHandlerMissingError:
            return {}

        preserved: dict[int, str] = {}

        for document in documents:
            binding = document.sync

            if document.id is None or binding is None or binding.storage_version is None:
                continue

            key = generate_storage_key(document.owner_id, document.group_id, document.id, binding.path)

            if await source.preserve(channel, binding.path, binding.storage_version, key):
                preserved[document.id] = key

        return preserved

    async def list_visible(self, actor: UserContext, space_id: int | None = None) -> list[ChannelView]:
        """Administrators see every channel; other users the channels they are responsible for."""
        async with self._uow_factory.create() as uow:
            channels = (
                await uow.sync_channels.list_by_space(space_id)
                if space_id is not None
                else await uow.sync_channels.list_all()
            )
            visible = [
                channel
                for channel in channels
                if channel.id is not None and channel.can_manage(actor.user_id, is_admin=actor.is_admin)
            ]
            latest = await uow.sync_runs.latest_for_channels([channel.id or 0 for channel in visible])

        return [ChannelView(channel, latest.get(channel.id or 0)) for channel in visible]

    async def adopt_ingested(self, channel_id: int, actor: UserContext) -> AdoptionReport:
        """Bind documents CLI ingestion made from an S3 channel's prefix to the channel.

        The prefix is read as a whole first, without a transaction, and selected exactly
        as a run selects it: a document whose file the channel would not publish stays
        manual (a run would delete it otherwise). Ids, access and vectors are kept. The
        published version is carried over only when this document provably indexes the
        file's current content; otherwise the next run re-indexes it. A document with
        manual chunk edits or other access stays manual. Repeating the call changes nothing.
        """
        if not actor.is_admin:
            msg = "Only an administrator can move documents into a sync channel"
            raise BusinessRuleViolationError(msg)

        async with self._uow_factory.create(master=True) as uow:
            channel = await existing_channel(uow, channel_id)

        if not isinstance(channel.settings.location, S3Location):
            msg = "Only an S3 channel can adopt ingested documents"
            raise BusinessRuleViolationError(msg)

        if self._sources is None:
            msg = "S3 channels are not available in this deployment"
            raise BusinessRuleViolationError(msg)

        # Raises StorageReadError when the prefix cannot be read: nothing is moved then.
        selection = await select_snapshot(self._sources.for_channel(channel), channel)
        report = AdoptionReport()

        async with self._uow_factory.create(master=True) as uow:
            current = await existing_channel(uow, channel_id)
            await lock_space(uow, current.space_id)
            current = await existing_channel(uow, channel_id, for_update=True)

            if current.generation != channel.generation:
                msg = "The channel changed while its storage was read; try again"
                raise BusinessRuleViolationError(msg)

            await self.adopt_selected(uow, current, selection, report)

        log.info("Sync channel %d adopted %d ingested documents", channel_id, len(report.adopted))
        return report

    @staticmethod
    async def adopt_selected(
        uow: UnitOfWork, channel: SyncChannel, selection: SnapshotSelection, report: AdoptionReport
    ) -> None:
        channel_id = channel.id or 0
        bound = {doc.sync.path for doc in await uow.documents.list_channel_documents(channel_id) if doc.sync}
        folder = channel.settings.folder
        candidates = [
            document
            for document in await uow.documents.list_ingestion_documents(f"{folder}/" if folder else "")
            if document.space_id == channel.space_id and document.sync is None and document.id is not None
        ]
        assignments = Counter(document.ingestion_key for document in candidates)

        for document in candidates:
            path = document.ingestion_key or ""
            entry = selection.desired.get(path)
            reason = (
                SKIP_UNREADABLE
                if path in selection.unreadable
                else SKIP_NOT_PUBLISHED
                if entry is None
                else SKIP_BOUND
                if path in bound
                else SKIP_MANUAL_EDITS
                if document.has_manual_edits
                else None
            )

            if reason is not None:
                report.skipped.append(SkippedPath(path, reason))
                continue

            if not audience_matches(channel, document):
                report.different_access.append(path)
                continue

            registry = await uow.ingestion_registry.get(path)
            # Fast path only for a version this very document published: one assignment
            # of the key, indexed, no revision pending, and the file unchanged since.
            proven = (
                entry is not None
                and registry is not None
                and registry.file_hash == entry.legacy_version
                and assignments[document.ingestion_key] == 1
                and document.status == DocumentStatus.DONE
                and await uow.chunks.get_staged_revision(document.id or 0) is None
            )
            binding = SyncBinding(
                channel_id,
                path,
                entry.version if proven and entry is not None else None,
                selection.revision if proven else None,
            )
            await uow.documents.set_sync_binding(document.id or 0, binding)
            bound.add(path)
            report.adopted.append(path)

    async def runs(self, channel_id: int, actor: UserContext, limit: int = 20) -> list[SyncRun]:
        async with self._uow_factory.create() as uow:
            channel = await existing_channel(uow, channel_id)

            if not channel.can_manage(actor.user_id, is_admin=actor.is_admin):
                msg = "Only an administrator or the channel's responsible user can see its runs"
                raise BusinessRuleViolationError(msg)

            return await uow.sync_runs.list_for_channel(channel_id, limit=limit)

    @staticmethod
    async def validate(uow: UnitOfWork, space: KnowledgeSpace, channel: SyncChannel) -> None:
        ensure_channel_scope(channel.settings)
        problem = await channel_rights_problem(uow, space, channel.settings)

        if problem is not None:
            msg = f"Sync channel cannot be saved: {problem}"
            raise BusinessRuleViolationError(msg)

        ensure_no_overlap(channel, await uow.sync_channels.list_by_space(channel.space_id))


UNLINKED_WITHOUT_SOURCE_WARNING = (
    "Исходный файл канала синхронизации не сохранён: документ нельзя переиндексировать"
)


async def free_filename(uow: UnitOfWork, document: Document) -> str:
    """The document's name, or the first free ``name(N).ext`` in the same folder of the path."""
    path = PurePosixPath(document.filename)
    candidate = document.filename
    counter = 1

    while (
        await uow.documents.find_active_slot(
            document.owner_id, candidate, document.group_id, space_id=document.space_id
        )
        is not None
    ):
        candidate = str(path.with_name(f"{path.stem}({counter}){path.suffix}"))
        counter += 1

    return candidate


async def unlink_document(uow: UnitOfWork, document: Document, source_path: str | None) -> None:
    """Turn a channel document into a manual one under a free name, keeping id and access."""
    document_id = document.id or 0
    await uow.documents.get_by_id(document_id, for_update=True)
    filename = await free_filename(uow, document)

    if filename != document.filename:
        await uow.documents.update_filename(document_id, filename, source_path or document.source_path)
        await uow.chunks.update_filename_by_document_id(document_id, filename)
        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.RENAME_DOCUMENT,
                aggregate_type="document",
                aggregate_id=document_id,
                payload={"document_id": document_id},
            )
        )

    await uow.documents.unlink_document(document_id, source_path)

    if source_path is None:
        # Status unchanged; the error is passed through, or the update would clear it.
        await uow.documents.update_status(
            document_id, warning=UNLINKED_WITHOUT_SOURCE_WARNING, error=document.error_message
        )


async def discard_staged_revision(uow: UnitOfWork, document_id: int) -> None:
    """Drop the document's unpublished revision and queue removal of its points."""
    await uow.documents.get_by_id(document_id, for_update=True)
    staged_ids = await uow.chunks.discard_staged_revision(document_id)

    if staged_ids:
        await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.DELETE_CHUNKS,
                aggregate_type="document",
                aggregate_id=document_id,
                payload={"chunk_ids": staged_ids},
            )
        )


async def lock_space(uow: UnitOfWork, space_id: int) -> KnowledgeSpace:
    space = await uow.knowledge_spaces.get_by_id(space_id, for_update=True)

    if space is None:
        msg = "Knowledge space"
        raise EntityNotFoundError(msg, space_id)

    return space


async def existing_channel(uow: UnitOfWork, channel_id: int, *, for_update: bool = False) -> SyncChannel:
    channel = await uow.sync_channels.get_by_id(channel_id, for_update=for_update)

    if channel is None:
        msg = "Sync channel"
        raise EntityNotFoundError(msg, channel_id)

    return channel
