"""SQLAlchemy persistence of document parent nodes using the caller's session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, insert

from infrastructure.database.models import DocumentNodeModel


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from domain.knowledge.document_nodes import DocumentNode


class SQLAlchemyDocumentNodeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def replace_document_nodes(
        self, document_id: int, act_version_id: int | None, nodes: list[DocumentNode]
    ) -> None:
        """Replace inside the chunk transaction; the caller holds the document chunk lock."""
        await self._session.execute(
            delete(DocumentNodeModel).where(DocumentNodeModel.document_id == document_id)
        )

        if not nodes:
            return

        await self._session.execute(
            insert(DocumentNodeModel),
            [
                {
                    "document_id": document_id,
                    "act_version_id": act_version_id,
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "kind": node.kind[:32],
                    "label": node.label,
                    "number": node.number[:64] if node.number else None,
                    "depth": node.depth,
                    "chunk_start": node.chunk_start,
                    "chunk_end": node.chunk_end,
                    "pages": list(node.pages),
                    "char_count": node.char_count,
                    "text_hash": node.text_hash,
                }
                for node in nodes
            ],
        )
