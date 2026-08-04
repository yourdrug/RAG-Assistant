"""ChatRAGPort — application-layer port for the RAG service used by ChatService.

Moved out of domain/repositories because the interface is shaped by
application orchestration (ACL context, depth) rather than domain concepts.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from domain.value_objects.chat_context import ChatContext


class ChatRAGPort(Protocol):
    async def stream(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> AsyncIterator[str]: ...

    async def invoke(
        self,
        question: str,
        history: list,
        ctx: ChatContext,
    ) -> tuple[str, list[dict]]: ...
