"""Chunks of a knowledge space shaped as retrieval returns them, read from PostgreSQL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from domain.benchmark.evaluation import source_name
from domain.knowledge.chunk_context import content_with_scope
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata


if TYPE_CHECKING:
    from application.uow import UnitOfWorkFactory


class UowBenchmarkChunkDocuments:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def space_documents(self, space_id: int, sources: set[str]) -> list[dict[str, Any]]:
        async with self._uow_factory.create() as uow:
            filenames = await uow.chunks.list_space_filenames(space_id)
            # File names are matched as the benchmark matches them: without directories and copy suffixes.
            document_ids = [doc_id for doc_id, name in filenames.items() if source_name(name) in sources]
            chunks = await uow.chunks.list_document_chunks(document_ids)

        documents = []

        for chunk in chunks:
            metadata = chunk_result_metadata(chunk)
            documents.append(
                {
                    "content": chunk.content,
                    "scoped_content": content_with_scope(metadata, chunk.content),
                    "metadata": metadata,
                }
            )

        return documents
