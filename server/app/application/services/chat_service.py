"""Application Service: ChatService — manages chat via UoWFactory.

Each method opens its own UnitOfWork. No db/session parameters.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from domain.entities.message import Message
from domain.repositories.rag_service_repository import RagServiceProtocol
from domain.value_objects.message_role import MessageRole
from infrastructure.uow_factory import UnitOfWorkFactory

from application.dto.chat_dto import ChatResult

log = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        rag_service: RagServiceProtocol,
        history_window: int = 8,
    ) -> None:
        self._uow_factory = uow_factory
        self._rag_service = rag_service
        self._history_window = history_window

    def _get_user_context(self, user_id: int, user_kind: str) -> tuple[list[int], list[int]]:
        """Fetch group_ids and assigned_client_ids for the user."""
        with self._uow_factory.create() as uow:
            if user_kind == "client":
                return [], []
            group_ids = uow.groups.get_user_group_ids(user_id)
            assigned_ids = uow.client_assignments.get_assigned_client_ids(user_id)
            return group_ids or [], assigned_ids or []

    async def stream_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
    ) -> AsyncIterator[str]:
        group_ids, assigned_ids = self._get_user_context(user_id, user_kind)

        with self._uow_factory.create() as uow:
            conv = uow.conversations.get_or_create(conversation_id, user_id)

            history = uow.messages.get_history(conv.id, window=self._history_window)
            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        full_answer = ""
        sources: list[dict] = []
        user_msg_saved = False

        async for chunk in self._rag_service.stream(
            question=question,
            history=history,
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=group_ids,
            assigned_client_ids=assigned_ids,
        ):
            # Save user message only after first real chunk arrives.
            # If user cancels before any chunks, the message is never saved.
            if not user_msg_saved and not chunk.startswith("\n__sources__:"):
                user_msg_saved = True
                with self._uow_factory.create() as uow:
                    user_msg = Message(
                        conversation_id=conv.id,
                        role=MessageRole.USER,
                        content=question,
                    )
                    uow.messages.save(user_msg)

            if chunk.startswith("\n__sources__:"):
                try:
                    sources = json.loads(chunk.replace("\n__sources__:", ""))
                except json.JSONDecodeError:
                    log.warning("Failed to parse sources chunk: %s", chunk)
            else:
                full_answer += chunk
                yield chunk

        with self._uow_factory.create() as uow:
            assistant_msg = Message(
                conversation_id=conv.id,
                role=MessageRole.ASSISTANT,
                content=full_answer,
                sources=sources,
            )
            uow.messages.save(assistant_msg)

        yield f"\n__meta__:{json.dumps({'conversation_id': conv.id, 'sources': sources}, ensure_ascii=False)}"

    async def sync_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
    ) -> ChatResult:
        group_ids, assigned_ids = self._get_user_context(user_id, user_kind)

        with self._uow_factory.create() as uow:
            conv = uow.conversations.get_or_create(conversation_id, user_id)

            if len(question.strip()) >= 3:
                user_msg = Message(
                    conversation_id=conv.id,
                    role=MessageRole.USER,
                    content=question,
                )
                uow.messages.save(user_msg)

            history = uow.messages.get_history(conv.id, window=self._history_window)
            if history and history[-1].role == MessageRole.USER:
                history = history[:-1]

        answer, sources = await self._rag_service.invoke(
            question=question,
            history=history,
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=group_ids,
            assigned_client_ids=assigned_ids,
        )

        with self._uow_factory.create() as uow:
            assistant_msg = Message(
                conversation_id=conv.id,
                role=MessageRole.ASSISTANT,
                content=answer,
                sources=sources,
            )
            uow.messages.save(assistant_msg)

        return ChatResult(answer=answer, conversation_id=conv.id, sources=sources)
