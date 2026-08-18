"""ChatRAGPort -- application-layer port for the RAG streaming service.

Defines the ``stream_answer`` protocol consumed by ``ChatService``.  Moved
out of ``domain/repositories`` because the interface is shaped by application
orchestration (ACL context, depth) rather than pure domain concepts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from domain.value_objects.chat_context import ChatContext
from domain.value_objects.stream_events import StreamEvent

from application.dto.chat_dto import RagResult


@runtime_checkable
class ChatRAGPort(Protocol):
    async def stream(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> AsyncIterator[StreamEvent]: ...

    async def invoke(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> RagResult: ...
