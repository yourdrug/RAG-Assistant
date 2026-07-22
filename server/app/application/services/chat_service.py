"""Application Service: ChatService — orchestrates chat with RAG."""

from __future__ import annotations

from collections.abc import AsyncIterator

from application.dto.chat_dto import ChatCommand, ChatResult
from application.use_cases.chat.stream_chat import StreamChat
from application.use_cases.chat.sync_chat import SyncChat


class ChatService:
    def __init__(
        self,
        stream_chat: StreamChat,
        sync_chat: SyncChat,
    ) -> None:
        self._stream_chat = stream_chat
        self._sync_chat = sync_chat

    async def stream_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> AsyncIterator[str]:
        async for chunk in self._stream_chat.execute(
            question=question,
            conversation_id=conversation_id,
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            user_group_ids=user_group_ids,
            assigned_client_ids=assigned_client_ids,
        ):
            yield chunk

    async def sync_chat(
        self,
        question: str,
        conversation_id: int | None,
        user_id: int,
        user_kind: str,
        user_role: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> ChatResult:
        return await self._sync_chat.execute(
            command=ChatCommand(question=question, conversation_id=conversation_id),
            user_id=user_id,
            user_kind=user_kind,
            user_role=user_role,
            user_group_ids=user_group_ids,
            assigned_client_ids=assigned_client_ids,
        )
