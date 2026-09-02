"""Application service for chunk lifecycle management.

Provides edit, add, delete, and list operations for document chunks.
Each public method opens its own async UnitOfWork via the injected UnitOfWorkFactory.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from domain.entities.document import Document
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.access_control import (
    compute_owner_and_group,
    validate_document_visibility,
)
from domain.utils import content_hash
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.visibility import DocumentVisibility

from application.dto.document_dto import DocumentDTO
from application.services.document_service import check_document_access, to_document_dto
from application.ports.chunk_settings import ChunkSettingsPort
from application.ports.unit_of_work_factory import UnitOfWorkFactory

if TYPE_CHECKING:
    from infrastructure.ml.client_registry import MLClientRegistry

log = logging.getLogger(__name__)


class ChunkService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
        chunk_settings: ChunkSettingsPort,
        chunk_min_len_ratio: float = 0.3,
        chunk_max_len_ratio: float = 2.0,
        ml_registry: MLClientRegistry | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store_repo
        self._settings = chunk_settings
        self._chunk_min_len_ratio = chunk_min_len_ratio
        self._chunk_max_len_ratio = chunk_max_len_ratio
        self._ml_registry = ml_registry

    async def list_chunks(
        self,
        document_id: int,
        user_id: int,
        user_kind: str,
        user_role: str,
        limit: int = 50,
        offset: int = 0,
        content_hashes: list[str] | None = None,
    ) -> tuple[list[dict], int]:
        """List chunks for a document with pagination."""
        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            await check_document_access(uow, doc, user_id, user_kind, user_role)

            chunks, total = await uow.chunks.list_for_document(
                document_id,
                limit=limit,
                offset=offset,
                content_hashes=content_hashes,
            )

            return [
                {
                    "id": c.chunk_id,
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
                    "content_hash": c.content_hash,
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

        async with self._uow_factory.create(master=True) as uow:
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

            self._validate_chunk_content(content, is_manual=(doc.source_type == "manual"))

            warning = await self._check_duplicate_content(uow, content, document_id, chunk_id)

            new_vector = await self._vector_store.generate_embeddings(content)

            new_hash = content_hash(content)
            existing_payload = await self._vector_store.get_point_payload(chunk_id)
            existing_metadata = (existing_payload or {}).get("metadata") or {}
            now = datetime.now(UTC)
            metadata = {
                **existing_metadata,
                "document_id": document_id,
                "visibility": doc.visibility.value if hasattr(doc.visibility, "value") else doc.visibility,
                "owner_id": doc.owner_id,
                "group_id": doc.group_id,
                "source": doc.filename,
                "content_hash": new_hash,
                "doc_domain": doc.doc_domain,
                "edited": True,
                "edited_at": now.isoformat(),
            }
            payload = {
                "page_content": content,
                "metadata": metadata,
            }

            await self._vector_store.upsert_point(
                point_id=chunk_id,
                vector=new_vector,
                payload=payload,
            )

            # --- Incremental BM25 update ---
            if self._ml_registry is not None and chunk.content_hash is not None:
                from infrastructure.ml.bm25_updater import bm25_replace

                bm25_replace(self._ml_registry, chunk.content_hash, content, new_hash=new_hash)

            await uow.chunks.update_content(
                chunk_id=chunk_id,
                content=content,
                edited_at=now,
                edited_by=user_id,
            )

            await uow.documents.set_has_manual_edits(document_id, True)
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

        async with self._uow_factory.create(master=True) as uow:
            doc = await uow.documents.get_by_id(document_id)
            if doc is None:
                raise EntityNotFound("Document", document_id)

            if doc.status != DocumentStatus.DONE:
                raise BusinessRuleViolation("Can only add chunks to documents with status 'done'")

            if not doc.can_edit_chunks(user_id, role):
                raise BusinessRuleViolation("No permission to add chunks for this document")

            self._validate_chunk_content(content, is_manual=(doc.source_type == "manual"))

            warning = await self._check_duplicate_content(uow, content, document_id)

            max_index = await uow.chunks.get_max_chunk_index(document_id)
            next_index = max_index + 1

            new_hash = content_hash(content)

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
                content_hash=new_hash,
            )
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

            vector = await self._vector_store.generate_embeddings(content)

            await self._vector_store.upsert_point(
                point_id=chunk_id,
                vector=vector,
                payload={"page_content": content, "metadata": metadata},
            )

            # --- Incremental BM25 update ---
            if self._ml_registry is not None:
                from infrastructure.ml.bm25_updater import bm25_add

                bm25_add(self._ml_registry, content, text_hash=new_hash)

            await uow.documents.set_has_manual_edits(document_id, True)
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

        async with self._uow_factory.create(master=True) as uow:
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

            await uow.chunks.delete_one(chunk_id)

            await self._vector_store.delete_by_ids([chunk_id])

            # --- Incremental BM25 update ---
            if self._ml_registry is not None and chunk.content_hash is not None:
                from infrastructure.ml.bm25_updater import bm25_remove

                bm25_remove(self._ml_registry, chunk.content_hash)

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

        async with self._uow_factory.create(master=True) as uow:
            user_group_ids = await uow.groups.get_user_group_ids(user_id)
            validate_document_visibility(vis, group_id, user_kind_enum, user_role_enum, user_group_ids)

            owner_id, effective_group_id = compute_owner_and_group(vis, group_id, user_id)

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

            return to_document_dto(saved_doc, chunks=0, chars=0, source_type="manual")

    def _validate_chunk_content(self, content: str, *, is_manual: bool = False) -> None:
        """Validate chunk content length."""
        if not content or not content.strip():
            raise ValidationError("Chunk content cannot be empty")

        chunk_size = self._settings.chunk_size
        min_ratio = 0.05 if is_manual else self._chunk_min_len_ratio
        min_len = int(min_ratio * chunk_size)
        max_len = int(self._chunk_max_len_ratio * chunk_size)

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
        new_hash = content_hash(content)
        duplicate = await uow.chunks.find_duplicate_by_hash(
            document_id=document_id,
            content_hash=new_hash,
            exclude_chunk_id=exclude_chunk_id,
        )
        if duplicate is not None:
            return f"Text matches existing chunk #{duplicate.chunk_id}"
        return None

    async def _update_document_stats(self, uow, document_id: int) -> None:
        """Update document chunks and chars counts."""
        stats = await uow.chunks.get_document_stats(document_id)
        await uow.documents.update_chunk_stats(document_id, stats.total_chunks, stats.total_chars)
