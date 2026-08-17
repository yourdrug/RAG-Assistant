"""Application service for document quality management."""

from __future__ import annotations

from application.ports.unit_of_work_factory import UnitOfWorkFactory


class QualityService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list_warned_documents(self):
        async with self._uow_factory.create() as uow:
            all_docs = await uow.documents.list_all()
            warned = [
                d
                for d in all_docs
                if d.warning_message or (d.quality_score is not None and d.quality_score > 0.3)
            ]
            warned.sort(key=lambda d: d.quality_score or 0.0, reverse=True)
            return warned

    async def get_document_source_path(self, document_id: int) -> str:
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                from domain.exceptions import EntityNotFound
                raise EntityNotFound("Document", document_id)
            if not doc.source_path:
                from domain.exceptions import ValidationError
                raise ValidationError(detail="Document has no source file", field="source_path")
            return doc.source_path
