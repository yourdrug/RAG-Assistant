"""Combined chunk persistence facade sharing one UnitOfWork transaction."""

from typing import Protocol, runtime_checkable

from domain.knowledge.repositories.chunk import (
    ChunkCorpusRepository,
    ChunkCrudRepository,
    ChunkPackMetadataRepository,
    DocumentNodeRepository,
)
from domain.rag.ports import ChunkRetrievalRepository, ChunkSearchPageRepository, DocumentOutlineRepository


@runtime_checkable
class ChunkRepository(
    ChunkCrudRepository,
    ChunkCorpusRepository,
    ChunkRetrievalRepository,
    ChunkSearchPageRepository,
    DocumentOutlineRepository,
    ChunkPackMetadataRepository,
    DocumentNodeRepository,
    Protocol,
):
    """The UoW exposes all chunk ports over the same session."""
