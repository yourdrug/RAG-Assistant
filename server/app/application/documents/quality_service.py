"""Application service for document quality management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.exceptions import EntityNotFound, ValidationError


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory
    from domain.entities.document import Document


class QualityService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list_warned_documents(self) -> list[Document]:
        async with self._uow_factory.create() as uow:
            return await uow.documents.list_warned(quality_threshold=0.3)

    async def get_document_source_path(self, document_id: int) -> str:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)

            if doc is None:
                entity_name = "Document"
                raise EntityNotFound(entity_name, document_id)

            if not doc.source_path:
                msg = "Document has no source file"
                raise ValidationError(msg, field="source_path")

            return doc.source_path
