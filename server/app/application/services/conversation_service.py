"""Application service for conversation management with ownership enforcement."""

from __future__ import annotations

from domain.exceptions import EntityNotFound, PermissionDeniedError
from domain.value_objects.roles import UserRole

from application.ports.unit_of_work_factory import UnitOfWorkFactory


class ConversationService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

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
