"""BM25 index port — abstract interface for in-memory BM25 index operations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BM25IndexPort(Protocol):
    """Protocol for BM25 index mutations (add, remove, replace)."""

    async def remove(self, content_hash: str) -> None: ...

    async def add(
        self,
        text: str,
        *,
        text_hash: str,
        visibility: str | None = None,
        owner_id: int | None = None,
        group_id: int | None = None,
    ) -> None: ...

    async def replace(
        self,
        old_hash: str,
        new_text: str,
        *,
        new_hash: str,
        visibility: str | None = None,
        owner_id: int | None = None,
        group_id: int | None = None,
    ) -> None: ...
