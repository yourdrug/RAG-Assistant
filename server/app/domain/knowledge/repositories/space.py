"""Knowledge space repository interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from domain.knowledge.space import KnowledgeSpace


@runtime_checkable
class KnowledgeSpaceRepository(Protocol):
    async def get_by_id(self, space_id: int, *, for_update: bool = False) -> KnowledgeSpace | None: ...
    async def list_all(self) -> list[KnowledgeSpace]: ...
    async def create(self, space: KnowledgeSpace) -> KnowledgeSpace:
        """Insert a space; a duplicate name raises ``UniqueConstraintError``."""
        ...

    async def update(self, space: KnowledgeSpace) -> KnowledgeSpace:
        """Persist the name and admitted groups; the pack never changes."""
        ...

    async def delete(self, space_id: int) -> None: ...
    async def has_documents(self, space_id: int) -> bool: ...
