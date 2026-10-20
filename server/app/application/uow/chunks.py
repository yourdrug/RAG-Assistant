"""Combined chunk persistence facade sharing one UnitOfWork transaction."""

from typing import Protocol, runtime_checkable

from domain.knowledge.repositories.chunk import (
    ChunkCorpusRepository,
    ChunkCrudRepository,
    ChunkVersioningRepository,
)
from domain.repositories.chunk_repository import ChunkRetrievalRepository, ChunkSearchPageRepository


@runtime_checkable
class ChunkRepository(
    ChunkCrudRepository,
    ChunkCorpusRepository,
    ChunkRetrievalRepository,
    ChunkSearchPageRepository,
    ChunkVersioningRepository,
    Protocol,
):
    """The UoW exposes all chunk ports over the same session."""
