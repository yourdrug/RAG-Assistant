"""Read ports for chunk retrieval."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, runtime_checkable

from domain.identity.user_context import UserContext
from domain.knowledge.chunk import TABLE_CONTEXT_MAX_CHUNKS, ChunkRecord
from domain.knowledge.enums import SearchMode


@runtime_checkable
class ChunkSearchRepository(Protocol):
    """Substring search over document chunks."""

    async def search_substring(
        self,
        query: str,
        user: UserContext,
        limit: int = 20,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        as_of_date: date | None = None,
    ) -> list[ChunkRecord]: ...


@runtime_checkable
class ChunkSearchPageRepository(Protocol):
    """Paginated API search, with the total restricted to the same access scope."""

    async def search_substring_page(
        self,
        query: str,
        user: UserContext,
        limit: int = 20,
        offset: int = 0,
        mode: str = SearchMode.EXACT.value,
        document_id: int | None = None,
        as_of_date: date | None = None,
    ) -> tuple[list[ChunkRecord], int]: ...


@runtime_checkable
class ChunkContextRepository(Protocol):
    """Fetch surrounding chunk context without bypassing request ACL."""

    async def get_neighbors(
        self,
        document_id: int,
        center_index: int,
        window: int = 1,
        exclude_hashes: set[str] | None = None,
        *,
        user: UserContext,
    ) -> list[ChunkRecord]: ...

    async def get_table_batches(
        self,
        document_id: int,
        table_id: str,
        exclude_hashes: set[str] | None = None,
        *,
        user: UserContext,
        limit: int = TABLE_CONTEXT_MAX_CHUNKS,
        row_start: int | None = None,
        row_end: int | None = None,
    ) -> list[ChunkRecord]: ...


@runtime_checkable
class ChunkRetrievalRepository(ChunkSearchRepository, ChunkContextRepository, Protocol):
    """Substring matches and surrounding context for RAG retrieval."""


@runtime_checkable
class VectorSearch(Protocol):
    async def generate_embeddings(self, text: str) -> list[float]: ...
    async def get_point_payload(self, point_id: int, access_filter: Any) -> dict | None: ...
