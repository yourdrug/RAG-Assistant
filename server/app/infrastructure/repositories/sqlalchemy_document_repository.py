"""SQLAlchemy ORM implementation of DocumentRepository."""

from __future__ import annotations

from datetime import UTC, datetime

from domain.entities.document import Document
from domain.services.access_control import get_visibility_conditions
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.owner_match import OwnerMatch
from domain.value_objects.roles import UserKind
from domain.value_objects.visibility import DocumentVisibility
from sqlalchemy import and_, distinct, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import DocumentModel


class SQLAlchemyDocumentRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save(self, document: Document) -> Document:
        orm = DocumentModel(
            filename=document.filename,
            source_path=document.source_path,
            visibility=document.visibility.value
            if hasattr(document.visibility, "value")
            else document.visibility,
            owner_id=document.owner_id,
            group_id=document.group_id,
            status=document.status.value if hasattr(document.status, "value") else document.status,
            doc_domain=document.doc_domain,
            source_type=document.source_type,
            has_manual_edits=document.has_manual_edits,
            chunks=document.chunks,
            chars=document.chars,
        )
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        document.id = orm.id
        return document

    async def get_by_id(self, document_id: int) -> Document | None:
        result = await self._db.execute(select(DocumentModel).where(DocumentModel.id == document_id))
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def delete(self, document_id: int) -> None:
        result = await self._db.execute(select(DocumentModel).where(DocumentModel.id == document_id))
        orm = result.scalar_one_or_none()
        if orm:
            await self._db.delete(orm)
            await self._db.flush()

    async def update_status(
        self,
        document_id: int,
        status: str,
        error: str | None = None,
        chunks: int | None = None,
        chars: int | None = None,
        warning: str | None = None,
        quality_score: float | None = None,
    ) -> None:
        result = await self._db.execute(select(DocumentModel).where(DocumentModel.id == document_id))
        orm = result.scalar_one_or_none()
        if orm is None:
            return

        orm.status = status
        orm.error_message = error
        if status == DocumentStatus.DONE.value:
            orm.warning_message = warning
        if quality_score is not None:
            orm.quality_score = quality_score
        if chunks is not None:
            orm.chunks = chunks
        if chars is not None:
            orm.chars = chars
        if status == DocumentStatus.DONE.value:
            orm.indexed_at = datetime.now(tz=UTC)
        await self._db.flush()

    async def set_source_path(self, document_id: int, source_path: str) -> None:
        result = await self._db.execute(select(DocumentModel).where(DocumentModel.id == document_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.source_path = source_path
            await self._db.flush()

    async def set_domain(self, document_id: int, doc_domain: str) -> None:
        result = await self._db.execute(select(DocumentModel).where(DocumentModel.id == document_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.doc_domain = doc_domain
            await self._db.flush()

    async def find_active_slot(
        self, owner_id: int | None, filename: str, group_id: int | None, for_update: bool = False
    ) -> Document | None:
        stmt = (
            select(DocumentModel)
            .where(
                DocumentModel.filename == filename,
                DocumentModel.owner_id == owner_id,
                DocumentModel.group_id == group_id,
                DocumentModel.status.in_(
                    [
                        DocumentStatus.PENDING.value,
                        DocumentStatus.PROCESSING.value,
                        DocumentStatus.DONE.value,
                        DocumentStatus.FAILED.value,
                    ]
                ),
            )
            .order_by(DocumentModel.creation_date.desc())
            .limit(1)
        )
        if for_update:
            stmt = stmt.with_for_update()
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def list_visible(
        self,
        user_kind: str,
        user_id: int,
        group_ids: list[int],
    ) -> list[Document]:
        conditions = get_visibility_conditions(
            UserKind(user_kind),
            user_id,
            group_ids,
        )

        or_clauses = []
        for cond in conditions:
            and_parts = [DocumentModel.visibility == cond.visibility.value]

            if cond.owner_match == OwnerMatch.SELF.value:
                and_parts.append(DocumentModel.owner_id == user_id)

            if cond.group_match:
                and_parts.append(DocumentModel.group_id.in_(group_ids))

            or_clauses.append(and_(*and_parts))

        if not or_clauses:
            return []

        result = await self._db.execute(
            select(DocumentModel).where(or_(*or_clauses)).order_by(DocumentModel.creation_date.desc())
        )
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def list_all(self) -> list[Document]:
        result = await self._db.execute(select(DocumentModel).order_by(DocumentModel.creation_date.desc()))
        return [self._to_entity(orm) for orm in result.scalars().all()]

    async def set_has_manual_edits(self, document_id: int, value: bool) -> None:
        result = await self._db.execute(select(DocumentModel).where(DocumentModel.id == document_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.has_manual_edits = value
            await self._db.flush()

    async def update_chunk_stats(self, document_id: int, chunks: int, chars: int) -> None:
        result = await self._db.execute(select(DocumentModel).where(DocumentModel.id == document_id))
        orm = result.scalar_one_or_none()
        if orm:
            orm.chunks = chunks
            orm.chars = chars
            await self._db.flush()

    async def list_distinct_filenames(self, search: str | None = None, limit: int = 100) -> list[str]:
        stmt = select(distinct(DocumentModel.filename)).where(
            DocumentModel.status == DocumentStatus.DONE.value
        )
        if search:
            stmt = stmt.where(DocumentModel.filename.ilike(f"%{search}%"))
        stmt = stmt.order_by(DocumentModel.filename).limit(limit)
        result = await self._db.execute(stmt)
        return [row[0] for row in result.all() if row[0]]

    async def delete_internal_documents(self) -> int:
        """Delete all documents with owner_id IS NULL (CLI-ingested + public UI docs).

        Manual documents (source_type='manual') are preserved.
        Returns the number of deleted documents.
        """
        result = await self._db.execute(
            select(DocumentModel).where(
                DocumentModel.owner_id.is_(None),
                DocumentModel.source_type != "manual",
            )
        )
        orms = result.scalars().all()
        for orm in orms:
            await self._db.delete(orm)
        await self._db.flush()
        return len(orms)

    @staticmethod
    def _to_entity(orm: DocumentModel) -> Document:
        return Document(
            id=orm.id,
            filename=orm.filename,
            source_path=orm.source_path,
            visibility=DocumentVisibility(orm.visibility),
            owner_id=orm.owner_id,
            group_id=orm.group_id,
            status=DocumentStatus(orm.status),
            doc_domain=orm.doc_domain,
            source_type=orm.source_type,
            has_manual_edits=orm.has_manual_edits,
            error_message=orm.error_message,
            warning_message=orm.warning_message,
            quality_score=orm.quality_score,
            chunks=orm.chunks,
            chars=orm.chars,
            creation_date=orm.creation_date,
            indexed_at=orm.indexed_at,
        )
