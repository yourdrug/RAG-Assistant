"""Application service for chunk lifecycle management.

Provides edit, add, delete, and list operations for document chunks.
Each public method opens its own async UnitOfWork via the injected UnitOfWorkFactory.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from config import settings
from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.access_control import can_view_document, validate_document_visibility
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.ml.hybrid import content_hash

from application.dto.document_dto import DocumentDTO
from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger(__name__)


class ChunkService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store_repo

    async def list_chunks(
        self,
        document_id: int,
        user_id: int,
        user_kind: str,
        user_role: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List chunks for a document with pagination."""
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            user_group_ids = (
                await uow.groups.get_user_group_ids(user_id) if user_kind == UserKind.INTERNAL else []
            )
            assigned_ids = (
                await uow.client_assignments.get_assigned_client_ids(user_id)
                if user_kind == UserKind.INTERNAL
                else []
            )

            if not can_view_document(
                doc_visibility=doc.visibility,
                doc_owner_id=doc.owner_id,
                doc_group_id=doc.group_id,
                user_kind=user_kind,
                user_id=user_id,
                user_group_ids=user_group_ids,
                assigned_client_ids=assigned_ids,
                user_role=user_role,
            ):
                raise BusinessRuleViolation("No access to this document")

            from infrastructure.database.models import ChunkModel
            from sqlalchemy import func, select

            # Get total count
            count_stmt = (
                select(func.count()).select_from(ChunkModel).where(ChunkModel.document_id == document_id)
            )
            total_result = await uow._session.execute(count_stmt)
            total = total_result.scalar() or 0

            # Get chunks with pagination
            stmt = (
                select(ChunkModel)
                .where(ChunkModel.document_id == document_id)
                .order_by(ChunkModel.chunk_index)
                .limit(limit)
                .offset(offset)
            )
            result = await uow._session.execute(stmt)
            chunks = result.scalars().all()

            return [
                {
                    "id": c.id,
                    "document_id": c.document_id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "filename": c.filename,
                    "visibility": c.visibility,
                    "doc_domain": c.doc_domain,
                    "owner_id": c.owner_id,
                    "group_id": c.group_id,
                    "edited_at": c.edited_at.isoformat() if c.edited_at else None,
                    "edited_by": c.edited_by,
                    "manual": c.manual,
                    "creation_date": c.creation_date.isoformat() if c.creation_date else None,
                }
                for c in chunks
            ], total

    async def edit_chunk(
        self,
        document_id: int,
        chunk_id: int,
        content: str,
        user_id: int,
        user_role: str,
    ) -> dict:
        """Edit an existing chunk's content with automatic re-embedding."""
        role = UserRole(user_role)

        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            if not doc.can_edit_chunks(user_id, role):
                raise BusinessRuleViolation("No permission to edit chunks for this document")

            chunk = await uow.chunks.get_by_id(chunk_id)
            if chunk is None:
                raise EntityNotFound("Chunk", chunk_id)

            if chunk.document_id != document_id:
                raise BusinessRuleViolation("Chunk does not belong to this document")

            # Validate content length
            self._validate_chunk_content(content)

            # Check for duplicate content
            warning = await self._check_duplicate_content(uow, content, document_id, chunk_id)

            # Generate new embedding
            new_vector = await self._vector_store.generate_embeddings(content)

            # Update Qdrant point (same point_id = chunk.id)
            new_hash = content_hash(content)
            payload = {
                "page_content": content,
                "metadata": {
                    "document_id": document_id,
                    "visibility": doc.visibility.value
                    if hasattr(doc.visibility, "value")
                    else doc.visibility,
                    "owner_id": doc.owner_id,
                    "group_id": doc.group_id,
                    "source": doc.filename,
                    "content_hash": new_hash,
                    "doc_domain": doc.doc_domain,
                    # Preserve page/section from existing chunk metadata if available
                },
            }

            # Try to get existing metadata from Qdrant to preserve page/section
            # For now, we'll use the basic metadata

            await self._vector_store.upsert_point(
                point_id=chunk_id,
                vector=new_vector,
                payload=payload,
            )

            # Update Postgres
            now = datetime.now(UTC)
            await uow.chunks.update_content(
                chunk_id=chunk_id,
                content=content,
                edited_at=now,
                edited_by=user_id,
            )

            # Update document has_manual_edits flag
            await self._set_document_has_manual_edits(uow, document_id, True)

            # Update document chars count
            await self._update_document_stats(uow, document_id)

            log.info(
                "Chunk %d edited by user %d in document %d",
                chunk_id,
                user_id,
                document_id,
            )

            result = {
                "id": chunk_id,
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "content": content,
                "edited_at": now.isoformat(),
                "edited_by": user_id,
                "manual": chunk.manual,
            }
            if warning:
                result["warning"] = warning
            return result

    async def add_chunk(
        self,
        document_id: int,
        content: str,
        user_id: int,
        user_role: str,
        page: int | None = None,
        section: str | None = None,
    ) -> dict:
        """Add a new chunk to an existing document."""
        role = UserRole(user_role)

        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            if doc.status != DocumentStatus.DONE:
                raise BusinessRuleViolation("Can only add chunks to documents with status 'done'")

            if not doc.can_edit_chunks(user_id, role):
                raise BusinessRuleViolation("No permission to add chunks for this document")

            # Validate content length
            self._validate_chunk_content(content)

            # Check for duplicate content
            warning = await self._check_duplicate_content(uow, content, document_id)

            # Get next chunk_index
            max_index = await uow.chunks.get_max_chunk_index(document_id)
            next_index = max_index + 1

            # Insert into Postgres
            chunk_id = await uow.chunks.insert_one(
                document_id=document_id,
                chunk_index=next_index,
                content=content,
                filename=doc.filename,
                visibility=doc.visibility.value if hasattr(doc.visibility, "value") else doc.visibility,
                doc_domain=doc.doc_domain,
                owner_id=doc.owner_id,
                group_id=doc.group_id,
                manual=True,
            )

            # Generate embedding
            vector = await self._vector_store.generate_embeddings(content)

            # Upsert to Qdrant with chunk.id as point_id
            new_hash = content_hash(content)
            metadata = {
                "document_id": document_id,
                "visibility": doc.visibility.value if hasattr(doc.visibility, "value") else doc.visibility,
                "owner_id": doc.owner_id,
                "group_id": doc.group_id,
                "source": doc.filename,
                "content_hash": new_hash,
                "doc_domain": doc.doc_domain,
                "manual": True,
            }
            if page is not None:
                metadata["page"] = page
            if section is not None:
                metadata["section"] = section

            await self._vector_store.upsert_point(
                point_id=chunk_id,
                vector=vector,
                payload={"page_content": content, "metadata": metadata},
            )

            # Update document stats
            await self._set_document_has_manual_edits(uow, document_id, True)
            await self._update_document_stats(uow, document_id)

            log.info(
                "Chunk %d added to document %d by user %d",
                chunk_id,
                document_id,
                user_id,
            )

            result = {
                "id": chunk_id,
                "document_id": document_id,
                "chunk_index": next_index,
                "content": content,
                "manual": True,
            }
            if warning:
                result["warning"] = warning
            return result

    async def delete_chunk(
        self,
        document_id: int,
        chunk_id: int,
        user_id: int,
        user_role: str,
    ) -> None:
        """Delete a single chunk."""
        role = UserRole(user_role)

        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            if not doc.can_edit_chunks(user_id, role):
                raise BusinessRuleViolation("No permission to delete chunks for this document")

            chunk = await uow.chunks.get_by_id(chunk_id)
            if chunk is None:
                raise EntityNotFound("Chunk", chunk_id)

            if chunk.document_id != document_id:
                raise BusinessRuleViolation("Chunk does not belong to this document")

            # Delete from Postgres
            await uow.chunks.delete_one(chunk_id)

            # Delete from Qdrant
            await self._vector_store.delete_by_ids([chunk_id])

            # Update document stats
            await self._update_document_stats(uow, document_id)

            log.info(
                "Chunk %d deleted from document %d by user %d",
                chunk_id,
                document_id,
                user_id,
            )

    async def create_manual_document(
        self,
        title: str,
        visibility: str,
        user_id: int,
        user_kind: str,
        user_role: str,
        group_id: int | None = None,
    ) -> DocumentDTO:
        """Create a virtual document container for manual chunks."""
        vis = DocumentVisibility.validate(visibility)
        user_kind_enum = UserKind(user_kind)
        user_role_enum = UserRole(user_role)

        async with self._uow_factory.create() as uow:
            user_group_ids = await uow.groups.get_user_group_ids(user_id)
            validate_document_visibility(vis, group_id, user_kind_enum, user_role_enum, user_group_ids)

            owner_id, effective_group_id = self._compute_owner_and_group(vis, group_id, user_id)

            doc = Document(
                filename=title,
                source_path="",
                visibility=vis,
                owner_id=owner_id,
                group_id=effective_group_id,
                status=DocumentStatus.DONE,
                source_type="manual",
                chunks=0,
                chars=0,
            )

            saved_doc = await uow.documents.save(doc)

            log.info(
                "Manual document %d created by user %d: %s",
                saved_doc.id,
                user_id,
                title,
            )

            return DocumentDTO(
                id=saved_doc.id,
                filename=saved_doc.filename,
                visibility=saved_doc.visibility,
                status=saved_doc.status,
                source_path=saved_doc.source_path,
                creation_date=saved_doc.creation_date,
                indexed_at=saved_doc.indexed_at,
                error_message=saved_doc.error_message,
                chunks=0,
                chars=0,
                owner_id=owner_id,
                group_id=effective_group_id,
                source_type="manual",
            )

    def _validate_chunk_content(self, content: str) -> None:
        """Validate chunk content length."""
        if not content or not content.strip():
            raise ValidationError("Chunk content cannot be empty")

        # Get chunk_size from settings (default 550)
        chunk_size = getattr(settings, "chunk_size", 550)
        min_len = int(0.3 * chunk_size)
        max_len = int(2.0 * chunk_size)

        if len(content) < min_len:
            raise ValidationError(
                f"Chunk content too short ({len(content)} chars). "
                f"Minimum approximately {min_len} chars. "
                f"Consider adding more content or merging with adjacent chunks."
            )

        if len(content) > max_len:
            raise ValidationError(
                f"Chunk content too long ({len(content)} chars). "
                f"Maximum approximately {max_len} chars. "
                f"Consider splitting into multiple chunks using separate POST requests."
            )

    async def _check_duplicate_content(
        self,
        uow,
        content: str,
        document_id: int,
        exclude_chunk_id: int | None = None,
    ) -> str | None:
        """Check for duplicate content hash. Returns warning message if duplicate found."""
        from infrastructure.database.models import ChunkModel
        from sqlalchemy import select

        new_hash = content_hash(content)

        # Check within the same document
        conditions = [ChunkModel.document_id == document_id]
        if exclude_chunk_id is not None:
            conditions.append(ChunkModel.id != exclude_chunk_id)

        stmt = select(ChunkModel).where(*conditions)
        result = await uow._session.execute(stmt)
        existing_chunks = result.scalars().all()

        for chunk in existing_chunks:
            if content_hash(chunk.content) == new_hash:
                return f"Text matches existing chunk #{chunk.id}"

        return None

    async def _set_document_has_manual_edits(self, uow, document_id: int, value: bool) -> None:
        """Set the has_manual_edits flag on a document."""
        from infrastructure.database.models import DocumentModel
        from sqlalchemy import select

        stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        result = await uow._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm:
            orm.has_manual_edits = value
            await uow._session.flush()

    async def _update_document_stats(self, uow, document_id: int) -> None:
        """Update document chunks and chars counts."""
        from infrastructure.database.models import ChunkModel
        from sqlalchemy import func, select

        # Count chunks and sum chars
        stmt = select(
            func.count().label("chunks"),
            func.coalesce(func.sum(func.length(ChunkModel.content)), 0).label("chars"),
        ).where(ChunkModel.document_id == document_id)
        result = await uow._session.execute(stmt)
        row = result.one()

        # Update document
        from infrastructure.database.models import DocumentModel

        doc_stmt = select(DocumentModel).where(DocumentModel.id == document_id)
        doc_result = await uow._session.execute(doc_stmt)
        doc_orm = doc_result.scalar_one_or_none()
        if doc_orm:
            doc_orm.chunks = row.chunks
            doc_orm.chars = row.chars
            await uow._session.flush()

    @staticmethod
    def _compute_owner_and_group(
        visibility: DocumentVisibility,
        group_id: int | None,
        user_id: int,
    ) -> tuple[int | None, int | None]:
        """Compute owner_id and group_id based on visibility."""
        if visibility == DocumentVisibility.INTERNAL_PUBLIC:
            return None, None
        if visibility == DocumentVisibility.INTERNAL_GROUP:
            return None, group_id
        return user_id, None
