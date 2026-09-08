"""Application service for document quality management."""

from __future__ import annotations

from domain.exceptions import EntityNotFound, ValidationError

from application.ports.unit_of_work_factory import UnitOfWorkFactory


class QualityService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list_warned_documents(self):
        async with self._uow_factory.create() as uow:
            return await uow.documents.list_warned(quality_threshold=0.3)

    async def get_document_source_path(self, document_id: int) -> str:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)
            if not doc.source_path:
                raise ValidationError("Document has no source file", field="source_path")
            return doc.source_path
