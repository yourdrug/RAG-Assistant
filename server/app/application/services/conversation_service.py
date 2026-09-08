"""Application service for conversation management with ownership enforcement."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from application.ports.chat_settings import ChatSettingsPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from domain.exceptions import EntityNotFound, PermissionDeniedError
from domain.value_objects.message_role import MessageRole
from domain.value_objects.roles import UserRole

if TYPE_CHECKING:
    from application.ports.chat_support import RollingSummaryUpdaterPort

log = logging.getLogger(__name__)


class ConversationService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        summary_updater: RollingSummaryUpdaterPort,
        chat_settings: ChatSettingsPort,
    ) -> None:
        self._uow_factory = uow_factory
        self._summary_updater = summary_updater
        self._settings = chat_settings
        self._background_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Managed background tasks
    # ------------------------------------------------------------------

    def _spawn_background(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def shutdown(self) -> None:
        for t in self._background_tasks:
            t.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def list_by_user(self, user_id: int, limit: int = 50, offset: int = 0):
        async with self._uow_factory.create() as uow:
            return await uow.conversations.list_by_user(user_id, limit=limit, offset=offset)

    async def create(self, user_id: int):
        async with self._uow_factory.create(master=True) as uow:
            return await uow.conversations.create(user_id)

    async def get_history(self, conversation_id: int, user_id: int, user_role: str):
        async with self._uow_factory.create() as uow:
            owner_id = await uow.conversations.get_owner_id(conversation_id)
            if owner_id is None:
                raise EntityNotFound("Conversation", conversation_id)
            if owner_id != user_id and user_role != UserRole.ADMIN:
                raise PermissionDeniedError()
            messages = await uow.messages.get_history(conversation_id, window=100)
            return messages

    # ------------------------------------------------------------------
    # Rolling summary
    # ------------------------------------------------------------------

    async def update_rolling_summary(self, conv_id: int, question: str, answer: str, history: list) -> None:
        if not self._settings.rolling_summary_enabled:
            return
        if len(history) < self._settings.history_window:
            return

        recent_turns = [
            {"role": MessageRole.USER, "content": question},
            {"role": MessageRole.ASSISTANT, "content": answer},
        ]

        async def _update() -> None:
            try:
                async with self._uow_factory.create(master=True) as uow:
                    conv = await uow.conversations.get_by_id(conv_id)
                    if conv is None:
                        return
                    existing = conv.summary

                new_summary = await self._summary_updater.update(existing, recent_turns)

                async with self._uow_factory.create(master=True) as uow:
                    await uow.conversations.update_summary(conv_id, new_summary)
            except Exception:
                log.exception("Failed to update rolling summary for conv %d", conv_id)

        self._spawn_background(_update())
