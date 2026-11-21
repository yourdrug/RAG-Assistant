"""Create, change and delete sync channels.

Only an administrator binds a storage location or widens a channel; its responsible
user may narrow it. Channel changes of one space are serialized on the space row, so
two concurrent changes can never create overlapping channels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import TYPE_CHECKING

from application.documents.delete import delete_document_in_uow
from application.sync.audience import apply_audience
from application.sync.rights import channel_rights_problem
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError
from domain.sync.channel import ChannelDeletionMode, SyncChannel
from domain.sync.ownership import ensure_no_overlap


if TYPE_CHECKING:
    from application.indexing.pack_hooks import DocumentHooksResolver
    from application.ports.cache_invalidator import CacheInvalidatorPort
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.identity.user_context import UserContext
    from domain.knowledge.space import KnowledgeSpace
    from domain.sync.channel import ChannelSettings
    from domain.sync.run import SyncRun


log = logging.getLogger("default")


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
    ) -> None:
        self._uow_factory = uow_factory
        self._pack_hooks = pack_hooks
        self._cache_invalidator = cache_invalidator

    async def create(self, space_id: int, settings: ChannelSettings, actor: UserContext) -> SyncChannel:
        if not actor.is_admin:
            msg = "Only an administrator can create a sync channel"
            raise BusinessRuleViolationError(msg)

        async with self._uow_factory.create(master=True) as uow:
            space = await lock_space(uow, space_id)
            channel = SyncChannel(space_id=space_id, settings=settings)
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
            channel.change_settings(settings, user_id=actor.user_id, is_admin=actor.is_admin)

            changed_acl: list[int] = []

            if channel.settings != previous:
                await self.validate(uow, space, channel)
                await uow.sync_channels.update(channel)

                if channel.settings.audience != previous.audience:
                    changed_acl = await self.apply_audience(uow, channel)

        if changed_acl and self._cache_invalidator is not None:
            # Answers cached for the previous audience must not be served anymore.
            await self._cache_invalidator.invalidate_by_document_ids(changed_acl, raise_on_error=True)

        return channel

    @staticmethod
    async def apply_audience(uow: UnitOfWork, channel: SyncChannel) -> list[int]:
        """Narrowed access applies at once; the next run rewrites the vectors."""
        changed = []

        for document in await uow.documents.list_channel_documents(channel.id or 0):
            if await apply_audience(uow, channel, document) and document.id is not None and document.sync:
                changed.append(document.id)
                # Forget the published version: the next run re-indexes the file, so the
                # vector payloads carry the new audience too (unchanged vectors are reused).
                await uow.documents.set_sync_binding(
                    document.id, replace(document.sync, storage_version=None)
                )

        return changed

    async def delete(self, channel_id: int, mode: ChannelDeletionMode, actor: UserContext) -> int:
        """Delete the channel with its documents, or unlink them into manual documents.

        Returns how many documents were deleted or unlinked.
        """
        if not actor.is_admin:
            msg = "Only an administrator can delete a sync channel"
            raise BusinessRuleViolationError(msg)

        async with self._uow_factory.create(master=True) as uow:
            channel = await existing_channel(uow, channel_id)
            await lock_space(uow, channel.space_id)
            await existing_channel(uow, channel_id, for_update=True)

            if mode == ChannelDeletionMode.UNLINK:
                affected = await uow.documents.unlink_channel_documents(channel_id)
            else:
                documents = await uow.documents.list_channel_documents(channel_id)

                for document in documents:
                    if document.id is None:
                        continue

                    hooks = (
                        await self._pack_hooks.for_document(uow, document)
                        if self._pack_hooks is not None
                        else None
                    )
                    await delete_document_in_uow(uow, document.id, hooks)

                affected = len(documents)

            await uow.sync_channels.delete(channel_id)

        log.info("Sync channel %d deleted (%s, %d documents)", channel_id, mode.value, affected)
        return affected

    async def list_visible(self, actor: UserContext, space_id: int | None = None) -> list[ChannelView]:
        """Administrators see every channel; other users the channels they are responsible for."""
        async with self._uow_factory.create() as uow:
            channels = (
                await uow.sync_channels.list_by_space(space_id)
                if space_id is not None
                else await uow.sync_channels.list_all()
            )
            views = []

            for channel in channels:
                if channel.id is None or not channel.can_manage(actor.user_id, is_admin=actor.is_admin):
                    continue

                runs = await uow.sync_runs.list_for_channel(channel.id, limit=1)
                views.append(ChannelView(channel, runs[0] if runs else None))

        return views

    async def runs(self, channel_id: int, actor: UserContext, limit: int = 20) -> list[SyncRun]:
        async with self._uow_factory.create() as uow:
            channel = await existing_channel(uow, channel_id)

            if not channel.can_manage(actor.user_id, is_admin=actor.is_admin):
                msg = "Only an administrator or the channel's responsible user can see its runs"
                raise BusinessRuleViolationError(msg)

            return await uow.sync_runs.list_for_channel(channel_id, limit=limit)

    @staticmethod
    async def validate(uow: UnitOfWork, space: KnowledgeSpace, channel: SyncChannel) -> None:
        problem = await channel_rights_problem(uow, space, channel.settings)

        if problem is not None:
            msg = f"Sync channel cannot be saved: {problem}"
            raise BusinessRuleViolationError(msg)

        ensure_no_overlap(channel, await uow.sync_channels.list_by_space(channel.space_id))


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
