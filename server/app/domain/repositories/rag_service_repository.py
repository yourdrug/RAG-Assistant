"""RAG Service Protocol — abstracts the RAG service for chat use cases."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class RagServiceProtocol(Protocol):
    async def stream(
        self,
        question: str,
        history: list,
        user_id: int,
        user_kind: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> AsyncIterator[str]: ...

    async def invoke(
        self,
        question: str,
        history: list,
        user_id: int,
        user_kind: str,
        user_group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> tuple[str, list[dict]]: ...
