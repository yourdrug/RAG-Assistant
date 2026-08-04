"""Application Service: ChatService — manages chat via UoWFactory.

Each method opens its own async UnitOfWork. No db/session parameters.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from domain.entities.message import Message
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.message_role import MessageRole
from domain.value_objects.roles import UserKind
from infrastructure.uow_factory import UnitOfWorkFactory

from application.dto.chat_dto import ChatResult
from application.services.chat_rag_port import ChatRAGPort

log = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        rag_service: ChatRAGPort,
        history_window: int = 8,
    ) -> None:
        self._uow_factory = uow_factory
        self._rag_service = rag_service
        self._history_window = history_window

    async def _get_user_context(self, user_id: int, user_kind: str) -> tuple[list[int], list[int]]:
        async with self._uow_factory.create() as uow:
            if user_kind == UserKind.CLIENT:
                return [], []
            group_ids = await uow.groups.get_user_group_ids(user_id)
            assigned_ids = await uow.client_assignments.get_assigned_client_ids(user_id)
            return group_ids or [], assigned_ids or []

    async def stream_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        depth: str | None = None,
    ) -> AsyncIterator[str]:
        group_ids, assigned_ids = await self._get_user_context(user_id, user_kind)
        ctx = ChatContext(
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=group_ids,
            assigned_client_ids=assigned_ids,
            depth=depth,
        )

        async with self._uow_factory.create() as uow:
            conv = await uow.conversations.get_or_create(conversation_id, user_id)

            history = await uow.messages.get_history(conv.id, window=self._history_window)
            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        full_answer = ""
        sources: list[dict] = []
        user_msg_saved = False

        async for chunk in self._rag_service.stream(
            question=question,
            history=history,
            ctx=ctx,
        ):
            if not user_msg_saved and not chunk.startswith("\n__sources__:"):
                user_msg_saved = True
                async with self._uow_factory.create() as uow:
                    user_msg = Message(
                        conversation_id=conv.id,
                        role=MessageRole.USER,
                        content=question,
                    )
                    await uow.messages.save(user_msg)

            if chunk.startswith("\n__sources__:"):
                try:
                    sources = json.loads(chunk.replace("\n__sources__:", ""))
                except json.JSONDecodeError:
                    log.warning("Failed to parse sources chunk: %s", chunk)
            else:
                full_answer += chunk
                yield chunk

        async with self._uow_factory.create() as uow:
            assistant_msg = Message(
                conversation_id=conv.id,
                role=MessageRole.ASSISTANT,
                content=full_answer,
                sources=sources,
            )
            await uow.messages.save(assistant_msg)

        yield f"\n__meta__:{json.dumps({'conversation_id': conv.id, 'sources': sources}, ensure_ascii=False)}"

    async def sync_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        depth: str | None = None,
    ) -> ChatResult:
        group_ids, assigned_ids = await self._get_user_context(user_id, user_kind)
        ctx = ChatContext(
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=group_ids,
            assigned_client_ids=assigned_ids,
            depth=depth,
        )

        async with self._uow_factory.create() as uow:
            conv = await uow.conversations.get_or_create(conversation_id, user_id)

            if len(question.strip()) >= 3:
                user_msg = Message(
                    conversation_id=conv.id,
                    role=MessageRole.USER,
                    content=question,
                )
                await uow.messages.save(user_msg)

            history = await uow.messages.get_history(conv.id, window=self._history_window)
            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        answer, sources = await self._rag_service.invoke(
            question=question,
            history=history,
            ctx=ctx,
        )

        async with self._uow_factory.create() as uow:
            assistant_msg = Message(
                conversation_id=conv.id,
                role=MessageRole.ASSISTANT,
                content=answer,
                sources=sources,
            )
            await uow.messages.save(assistant_msg)

        return ChatResult(answer=answer, conversation_id=conv.id, sources=sources)
