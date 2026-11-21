"""Create, change and delete sync channels.

Only an administrator binds a storage location or widens a channel; its responsible
user may narrow it. Channel changes of one space are serialized on the space row, so
two concurrent changes can never create overlapping channels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.documents.delete import delete_document_in_uow
from application.sync.rights import channel_rights_problem
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError
from domain.sync.channel import ChannelDeletionMode, SyncChannel
from domain.sync.ownership import ensure_no_overlap


if TYPE_CHECKING:
    from application.indexing.pack_hooks import DocumentHooksResolver
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.identity.user_context import UserContext
    from domain.knowledge.space import KnowledgeSpace
    from domain.sync.channel import ChannelSettings


log = logging.getLogger("default")


class SyncChannelService:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, pack_hooks: DocumentHooksResolver | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._pack_hooks = pack_hooks

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

            if channel.settings != previous:
                await self.validate(uow, space, channel)
                await uow.sync_channels.update(channel)

        return channel

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
