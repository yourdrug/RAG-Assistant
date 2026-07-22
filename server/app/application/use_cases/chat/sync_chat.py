"""Use Case: SyncChat — synchronous RAG chat."""

from __future__ import annotations

from domain.entities.message import Message
from domain.repositories.conversation_repository import ConversationRepository
from domain.repositories.message_repository import MessageRepository
from domain.repositories.rag_service_repository import RagServiceProtocol
from domain.repositories.settings_repository import SettingsProtocol
from domain.value_objects.message_role import MessageRole

from application.dto.chat_dto import ChatCommand, ChatResult


class SyncChat:
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
        command: ChatCommand,
        user_id: int,
        user_kind: str,
        user_role: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> ChatResult:
        conv = self._conversation_repo.get_or_create(command.conversation_id, user_id)

        user_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content=command.question,
        )
        self._message_repo.save(user_msg)

        history = self._message_repo.get_history(conv.id, window=self._settings.history_window)
        if history and history[-1].role == MessageRole.USER:
            history = history[:-1]

        answer, sources = await self._rag_service.invoke(
            question=command.question,
            history=history,
            user_id=user_id,
            user_kind=user_kind,
            user_group_ids=user_group_ids,
            assigned_client_ids=assigned_client_ids,
        )

        assistant_msg = Message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content=answer,
            sources=sources,
        )
        self._message_repo.save(assistant_msg)

        return ChatResult(answer=answer, conversation_id=conv.id, sources=sources)
