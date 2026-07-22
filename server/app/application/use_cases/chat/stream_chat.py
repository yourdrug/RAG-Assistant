"""Use Case: StreamChat — streaming RAG chat."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from domain.entities.message import Message
from domain.repositories.conversation_repository import ConversationRepository
from domain.repositories.message_repository import MessageRepository
from domain.repositories.rag_service_repository import RagServiceProtocol
from domain.repositories.settings_repository import SettingsProtocol
from domain.value_objects.message_role import MessageRole

log = logging.getLogger(__name__)


class StreamChat:
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        rag_service: RagServiceProtocol,
        settings: SettingsProtocol,
    ) -> None:
        self._conversation_repo = conversation_repo
        self._message_repo = message_repo
        self._rag_service = rag_service
        self._settings = settings

    async def execute(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> AsyncIterator[str]:
        conv = self._conversation_repo.get_or_create(conversation_id, user_id)

        user_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content=question,
        )
        self._message_repo.save(user_msg)

        history = self._message_repo.get_history(conv.id, window=self._settings.history_window)
        if history and history[-1].role == MessageRole.USER:
            history = history[:-1]

        full_answer = ""
        sources: list[dict] = []

        async for chunk in self._rag_service.stream(
            question=question,
            history=history,
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=user_group_ids,
            assigned_client_ids=assigned_client_ids,
        ):
            if chunk.startswith("\n__sources__:"):
                try:
                    sources = json.loads(chunk.replace("\n__sources__:", ""))
                except json.JSONDecodeError:
                    log.warning("Failed to parse sources chunk: %s", chunk)
            else:
                full_answer += chunk
                yield chunk

        assistant_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=full_answer,
            sources=sources,
        )
        self._message_repo.save(assistant_msg)

        yield f"\n__meta__:{json.dumps({'conversation_id': conv.id, 'sources': sources}, ensure_ascii=False)}"
