"""ChunkMutationService -- edit, add, delete chunk operations with outbox + BM25."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from application.dto.chunk_dto import AddChunkResult, EditChunkResult
from application.services.chunk_access import load_doc_for_add, load_doc_for_edit
from application.services.document_pipeline import build_outbox_metadata
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.repositories.chunk_repository import ChunkCrudRepository
from domain.utils import content_hash
from domain.value_objects.source_type import SourceType
from domain.value_objects.chunk_context import extract_chunk_context

if TYPE_CHECKING:
    from application.ports.bm25_index import BM25IndexPort
    from application.ports.chunk_settings import ChunkSettingsPort
    from application.ports.unit_of_work_factory import UnitOfWorkFactory

log = logging.getLogger(__name__)


class ChunkMutationService:
    """Edit, add, delete chunk operations with outbox + BM25 side-effects."""

    def __init__(
        self,
        uow_factory: "UnitOfWorkFactory",
        chunk_settings: "ChunkSettingsPort",
        bm25_index: "BM25IndexPort",
        chunk_min_len_ratio: float = 0.3,
        chunk_max_len_ratio: float = 2.0,
    ) -> None:
        self._uow_factory = uow_factory
        self._settings = chunk_settings
        self._bm25_index = bm25_index
        self._chunk_min_len_ratio = chunk_min_len_ratio
        self._chunk_max_len_ratio = chunk_max_len_ratio

    def _validate_chunk_content(self, content: str, *, is_manual: bool = False) -> None:
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
        chunks: ChunkCrudRepository = uow.chunks
        new_hash = content_hash(content)
        duplicate = await chunks.find_duplicate_by_hash(
            document_id=document_id,
            content_hash=new_hash,
            exclude_chunk_id=exclude_chunk_id,
        )
        if duplicate is not None:
            return f"Text matches existing chunk #{duplicate.chunk_id}"
        return None

    async def _update_document_stats(self, uow, document_id: int) -> None:
        chunks: ChunkCrudRepository = uow.chunks
        stats = await chunks.get_document_stats(document_id)
        await uow.documents.update_chunk_stats(document_id, stats.total_chunks, stats.total_chars)

    async def edit_chunk(
        self,
        document_id: int,
        chunk_id: int,
        content: str,
        user_id: int,
        user_role: str,
    ) -> EditChunkResult:
        async with self._uow_factory.create(master=True) as uow:
            chunks: ChunkCrudRepository = uow.chunks
            doc, ctx = await load_doc_for_edit(uow, document_id, user_id, user_role)

            chunk = await chunks.get_by_id(chunk_id)
            if chunk is None:
                raise EntityNotFound("Chunk", chunk_id)

            if chunk.document_id != document_id:
                raise BusinessRuleViolation("Chunk does not belong to this document")

            self._validate_chunk_content(content, is_manual=(doc.source_type == SourceType.MANUAL.value))

            warning = await self._check_duplicate_content(uow, content, document_id, chunk_id)

            new_hash = content_hash(content)
            now = datetime.now(UTC)

            await chunks.update_content(
                chunk_id=chunk_id,
                content=content,
                edited_at=now,
                edited_by=user_id,
            )

            await uow.documents.set_has_manual_edits(document_id, True)
            await self._update_document_stats(uow, document_id)

            metadata = build_outbox_metadata(
                document_id=document_id,
                visibility=doc.visibility,
                owner_id=doc.owner_id,
                group_id=doc.group_id,
                filename=doc.filename,
                doc_domain=doc.doc_domain,
                content_hash=new_hash,
                edited=True,
                edited_at=now.isoformat(),
                chunk_index=chunk.chunk_index,
                act_version_id=chunk.act_version_id,
                effective_from=chunk.effective_from.isoformat() if chunk.effective_from else None,
                effective_to=chunk.effective_to.isoformat() if chunk.effective_to else None,
                is_current=chunk.is_current,
                **extract_chunk_context(chunk.context_metadata),
            )
            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.UPSERT_CHUNKS,
                    aggregate_type="document",
                    aggregate_id=document_id,
                    payload={
                        "points": [
                            {
                                "chunk_id": chunk_id,
                                "page_content": content,
                                "metadata": metadata,
                            }
                        ]
                    },
                )
            )

            log.info(
                "Chunk %d edited by user %d in document %d",
                chunk_id,
                user_id,
                document_id,
            )

            result = EditChunkResult(
                id=chunk_id,
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                content=content,
                edited_at=now.isoformat(),
                edited_by=user_id,
                manual=chunk.manual,
                warning=warning,
            )

        if chunk.content_hash is not None:
            vis = doc.visibility.value if hasattr(doc.visibility, "value") else doc.visibility
            self._bm25_index.replace(
                chunk.content_hash,
                content,
                new_hash=new_hash,
                visibility=vis,
                owner_id=doc.owner_id,
                group_id=doc.group_id,
            )

        return result

    async def add_chunk(
        self,
        document_id: int,
        content: str,
        user_id: int,
        user_role: str,
        page: int | None = None,
        section: str | None = None,
    ) -> AddChunkResult:
        async with self._uow_factory.create(master=True) as uow:
            chunks: ChunkCrudRepository = uow.chunks
            doc, ctx = await load_doc_for_add(uow, document_id, user_id, user_role)

            self._validate_chunk_content(content, is_manual=(doc.source_type == SourceType.MANUAL.value))

            warning = await self._check_duplicate_content(uow, content, document_id)

            max_index = await chunks.get_max_chunk_index(document_id)
            next_index = max_index + 1

            new_hash = content_hash(content)

            context_metadata = {}
            if page is not None:
                context_metadata["page"] = page
            if section is not None:
                context_metadata["section"] = section

            chunk_id = await chunks.insert_one(
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
                context_metadata=context_metadata,
            )
            metadata = build_outbox_metadata(
                document_id=document_id,
                visibility=doc.visibility,
                owner_id=doc.owner_id,
                group_id=doc.group_id,
                filename=doc.filename,
                doc_domain=doc.doc_domain,
                content_hash=new_hash,
                manual=True,
            )
            if page is not None:
                metadata["page"] = page
            if section is not None:
                metadata["section"] = section

            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.UPSERT_CHUNKS,
                    aggregate_type="document",
                    aggregate_id=document_id,
                    payload={
                        "points": [
                            {
                                "chunk_id": chunk_id,
                                "page_content": content,
                                "metadata": metadata,
                            }
                        ]
                    },
                )
            )

            await uow.documents.set_has_manual_edits(document_id, True)
            await self._update_document_stats(uow, document_id)

            log.info(
                "Chunk %d added to document %d by user %d",
                chunk_id,
                document_id,
                user_id,
            )

            result = AddChunkResult(
                id=chunk_id,
                document_id=document_id,
                chunk_index=next_index,
                content=content,
                manual=True,
                warning=warning,
            )

        vis = doc.visibility.value if hasattr(doc.visibility, "value") else doc.visibility
        self._bm25_index.add(
            content,
            text_hash=new_hash,
            visibility=vis,
            owner_id=doc.owner_id,
            group_id=doc.group_id,
        )

        return result

    async def delete_chunk(
        self,
        document_id: int,
        chunk_id: int,
        user_id: int,
        user_role: str,
    ) -> None:
        async with self._uow_factory.create(master=True) as uow:
            chunks: ChunkCrudRepository = uow.chunks
            doc, ctx = await load_doc_for_edit(uow, document_id, user_id, user_role)

            chunk = await chunks.get_by_id(chunk_id)
            if chunk is None:
                raise EntityNotFound("Chunk", chunk_id)

            if chunk.document_id != document_id:
                raise BusinessRuleViolation("Chunk does not belong to this document")

            await chunks.delete_one(chunk_id)

            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.DELETE_CHUNKS,
                    aggregate_type="document",
                    aggregate_id=document_id,
                    payload={"chunk_ids": [chunk_id]},
                )
            )

            await self._update_document_stats(uow, document_id)

            log.info(
                "Chunk %d deleted from document %d by user %d",
                chunk_id,
                document_id,
                user_id,
            )

        if chunk.content_hash is not None:
            self._bm25_index.remove(chunk.content_hash)
