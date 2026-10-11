"""DocumentSyncService -- two-phase DB sync for CLI ingestion.

Phase 1: resolve/create document rows (single committed transaction).
Phase 2: per document -- versioning (own tx) + chunks + outbox.

Extracted from IngestionService to isolate the Postgres sync concern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from application.dto.versioning_dto import VersioningResult
from application.services.document_pipeline import enrich_chunks_metadata, process_chunks, tag_chunks
from application.services.document_utils import resolve_unique_filename
from application.services.ingestion_source import object_key_from_source
from domain.entities.document import Document as DocEntity
from domain.entities.raw_document import RawDocument
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.exceptions import EntityNotFound, UniqueConstraintViolation, ValidationError
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.roles import UserKind
from domain.value_objects.visibility import DocumentVisibility

if TYPE_CHECKING:
    from domain.domain_profile.settings_port import DomainSettingsPort
    from application.ports.unit_of_work_factory import UnitOfWorkFactory
    from application.services.act_versioning_service import ActVersioningService
    from domain.domain_profile.registry import DomainProfileRegistry

log = logging.getLogger("default")
INGESTION_TRANSACTION_MAX_ATTEMPTS = 3


class DocumentSyncService:
    """Syncs documents and chunks to Postgres, enqueues outbox for Qdrant."""

    def __init__(
        self,
        uow_factory: "UnitOfWorkFactory",
        act_versioning_service: "ActVersioningService | None" = None,
        domain_registry: "DomainProfileRegistry | None" = None,
        domain_settings: "DomainSettingsPort | None" = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._act_versioning_service = act_versioning_service
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings

    def get_profile(self, domain: str):
        if self._domain_registry is None:
            return None
        try:
            return self._domain_registry.get(domain)
        except KeyError:
            return None

    async def sync_documents_to_db(
        self,
        registry: dict,
        source_chars: dict,
        chunks: list,
        full_text_by_source: dict[str, str] | None = None,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None:
        """Sync documents and chunks to Postgres, enqueue outbox for Qdrant.

        Two phases so that act-version rows (created in their own transaction,
        exactly like the API upload path) can reference committed documents:
          Phase 1: resolve/create document rows (committed).
          Phase 2: per document -- versioning (own tx) + chunks + outbox.
        """
        full_text_by_source = full_text_by_source or {}
        visibility = DocumentVisibility.validate(visibility)
        if visibility == DocumentVisibility.CLIENT_PRIVATE and client_id is None:
            raise ValidationError("client_id required for client_private ingestion")
        if visibility == DocumentVisibility.INTERNAL_GROUP and group_id is None:
            raise ValidationError("group_id required for internal_group ingestion")
        owner_id = client_id if visibility == DocumentVisibility.CLIENT_PRIVATE else None
        group_id = group_id if visibility == DocumentVisibility.INTERNAL_GROUP else None

        # Group chunks by source for per-document processing
        chunks_by_source: dict[str, list] = {}
        for c in chunks:
            src = c.metadata.get("source", "")
            chunks_by_source.setdefault(src, []).append(c)
        documents_to_sync = {
            fname: info for fname, info in registry.items() if chunks_by_source.get(info.get("source", ""))
        }

        # --- Phase 1: document rows (single committed transaction) ---
        documents = await self.ensure_document_rows(
            documents_to_sync, visibility=visibility, group_id=group_id, owner_id=owner_id
        )

        # --- Phase 2: versioning + chunks + outbox (per document) ---
        for object_key, info in documents_to_sync.items():
            document = documents[object_key]
            doc_id = document.id
            if doc_id is None:
                raise RuntimeError(f"Document {object_key} has no ID after creation")
            src = info.get("source", "")
            file_chunks = chunks_by_source.get(src, [])
            if not file_chunks:
                continue
            await self.sync_one_document(
                fname=document.filename,
                doc_id=doc_id,
                src=src,
                file_chunks=file_chunks,
                full_text=full_text_by_source.get(src, ""),
                visibility=visibility,
                owner_id=owner_id,
                group_id=group_id,
            )

        log.info("Synced %d documents to database via outbox", len(documents_to_sync))

    async def ensure_document_rows(
        self,
        registry: dict,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        owner_id: int | None = None,
    ) -> dict[str, DocEntity]:
        """Commit source bindings before content sync; retry creation races."""
        for attempt in range(INGESTION_TRANSACTION_MAX_ATTEMPTS):
            try:
                return await self.create_document_rows(registry, visibility, group_id, owner_id)
            except UniqueConstraintViolation:
                if attempt == INGESTION_TRANSACTION_MAX_ATTEMPTS - 1:
                    raise
                log.warning("Concurrent ingestion document creation; retrying transaction")
        raise RuntimeError("Ingestion document creation attempts exhausted")

    async def create_document_rows(
        self,
        registry: dict,
        visibility: DocumentVisibility,
        group_id: int | None,
        owner_id: int | None,
    ) -> dict[str, DocEntity]:
        documents: dict[str, DocEntity] = {}
        async with self._uow_factory.create(master=True) as uow:
            if visibility == DocumentVisibility.CLIENT_PRIVATE:
                if owner_id is None:
                    raise ValidationError("client_id required for client_private ingestion")
                client_user = await uow.users.get_by_id(owner_id)
                if client_user is None or client_user.kind != UserKind.CLIENT:
                    raise ValidationError("client_id must be a user with kind='client'")

            for object_key, info in registry.items():
                # Metadata source remains the full URI. Use its exact object key
                # even when a legacy in-memory caller still keys its dict by name.
                ingestion_key = object_key_from_source(info["source"])
                existing = await uow.documents.find_by_ingestion_key(
                    owner_id=owner_id,
                    ingestion_key=ingestion_key,
                    group_id=group_id,
                )
                if existing is None and owner_id is not None:
                    # Preserve the ID when privatizing a legacy ownerless ingestion row.
                    existing = await uow.documents.find_by_ingestion_key(
                        owner_id=None, ingestion_key=ingestion_key, group_id=group_id
                    )
                if existing:
                    if existing.id is None:
                        raise RuntimeError(f"Document {object_key} has no ID after source lookup")
                    documents[object_key] = existing
                else:
                    filename = await resolve_unique_filename(
                        uow.documents, owner_id, group_id, Path(ingestion_key).name
                    )
                    doc = DocEntity(
                        filename=filename,
                        ingestion_key=ingestion_key,
                        visibility=visibility,
                        owner_id=owner_id,
                        group_id=group_id,
                    )
                    saved = await uow.documents.save(doc)
                    if saved.id is None:
                        raise RuntimeError(f"Document {object_key} has no ID after save")
                    documents[object_key] = saved
        return documents

    async def sync_one_document(
        self,
        fname: str,
        doc_id: int,
        src: str,
        file_chunks: list,
        full_text: str,
        visibility: DocumentVisibility,
        owner_id: int | None,
        group_id: int | None,
    ) -> None:
        """Persist version state, chunks and outbox together for one document."""
        first_chunk = file_chunks[0]
        chunk_domain = first_chunk.metadata.get("doc_domain", DocDomain.GENERAL.value)

        versioning = VersioningResult(
            domain_metadata=None,
            act_version_id=None,
            act_id=None,
            effective_from=None,
            warning=None,
        )
        profile = self.get_profile(chunk_domain)
        versioning_plan = None
        act_versioning_service = self._act_versioning_service
        if act_versioning_service is not None and profile is not None:
            prepare = getattr(act_versioning_service, "prepare_document_versioning", None)
            if prepare is not None:
                versioning_plan = await prepare(profile, full_text)
                if versioning_plan is not None:
                    versioning = VersioningResult(
                        versioning_plan.domain_metadata,
                        None,
                        None,
                        versioning_plan.effective_from,
                        versioning_plan.warning,
                    )
            else:
                versioning = await act_versioning_service.process_document_versioning(
                    profile, doc_id, full_text
                )
            if versioning.warning:
                log.warning("Versioning warning for %s: %s", fname, versioning.warning)

        # The request scope owns ACL; parser metadata must not override it.
        # Keep the source chunks consistent for the subsequent BM25 projection.
        tag_chunks(
            file_chunks,
            visibility=visibility,
            owner_id=owner_id,
            group_id=group_id,
            client_id=owner_id if visibility == DocumentVisibility.CLIENT_PRIVATE else None,
        )
        raw_chunks = [
            RawDocument(page_content=c.page_content, metadata=dict(c.metadata)) for c in file_chunks
        ]

        for attempt in range(INGESTION_TRANSACTION_MAX_ATTEMPTS):
            try:
                versioning = await self.persist_source_document(
                    doc_id,
                    src,
                    raw_chunks,
                    visibility,
                    owner_id,
                    group_id,
                    chunk_domain,
                    profile,
                    versioning_plan,
                    versioning,
                )
                break
            except UniqueConstraintViolation:
                if attempt == INGESTION_TRANSACTION_MAX_ATTEMPTS - 1:
                    raise
                log.warning("Concurrent ingestion filename reservation; retrying document transaction")
        if act_versioning_service is not None:
            if versioning.act_id is not None:
                await act_versioning_service.invalidate_act_answers(versioning.act_id)
            else:
                await act_versioning_service.invalidate_document_answers(doc_id)

    async def persist_source_document(
        self,
        doc_id: int,
        src: str,
        raw_chunks: list[RawDocument],
        visibility: DocumentVisibility,
        owner_id: int | None,
        group_id: int | None,
        chunk_domain: str,
        profile,
        versioning_plan,
        versioning: VersioningResult,
    ) -> VersioningResult:
        """Commit display name, ACL, versioning, chunks and outbox together."""
        act_versioning_service = self._act_versioning_service
        async with self._uow_factory.create(master=True) as uow:
            document = await uow.documents.get_by_id(doc_id, for_update=True)
            if document is None:
                raise EntityNotFound("Document", doc_id)
            fname = document.filename
            occupied = await uow.documents.find_active_slot(owner_id, fname, group_id)
            if occupied is not None and occupied.id != doc_id:
                fname = await resolve_unique_filename(uow.documents, owner_id, group_id, fname)
            # Update filename and ACL atomically: an alias may be occupied in
            # the old scope but free in the destination scope (or vice versa).
            await uow.documents.set_acl(doc_id, visibility, owner_id, group_id, filename=fname)
            if versioning_plan is not None and versioning_plan.warning is None:
                if act_versioning_service is None:
                    raise RuntimeError("Versioning plan exists without an act versioning service")
                act_version = await act_versioning_service.create_version_in_uow(
                    uow,
                    profile,
                    doc_id,
                    versioning_plan.extracted_refs,
                    versioning_plan.effective_from,
                    versioning_plan.date_confidence,
                )
                versioning = VersioningResult(
                    versioning_plan.domain_metadata,
                    act_version.id,
                    act_version.act_id,
                    act_version.effective_from,
                    None,
                    effective_to=act_version.effective_to,
                    is_current=act_version.is_current,
                )

            enrich_chunks_metadata(
                raw_chunks,
                doc_id,
                visibility.value,
                owner_id,
                group_id,
                chunk_domain,
                domain_metadata=versioning.domain_metadata,
                act_version_id=versioning.act_version_id,
                act_id=versioning.act_id,
                effective_from=versioning.effective_from,
                effective_to=versioning.effective_to,
                is_current=versioning.is_current,
            )

            if versioning.warning:
                await uow.documents.update_status(
                    doc_id,
                    status=None,
                    warning=versioning.warning,
                )

            await process_chunks(
                uow_factory=self._uow_factory,
                document_id=doc_id,
                filename=fname,
                chunks=raw_chunks,
                visibility=visibility.value,
                owner_id=owner_id,
                group_id=group_id,
                doc_domain=chunk_domain,
                set_indexing=False,  # CLI: done immediately (outbox async)
                _existing_uow=uow,
            )

            await uow.vector_outbox.enqueue(
                VectorOutboxEntry(
                    operation=OutboxOperation.SET_DOCUMENT_ID,
                    aggregate_type="document",
                    aggregate_id=doc_id,
                    payload={"source": src, "document_id": doc_id},
                )
            )
        return versioning

    async def delete_internal_documents(self) -> None:
        """Delete all internal documents (owner_id=NULL, non-manual) from DB.

        Called during reset to remove stale document records whose vectors
        have already been deleted from Qdrant.
        """
        async with self._uow_factory.create(master=True) as uow:
            deleted = await uow.documents.delete_internal_documents()
            log.info("Deleted %d internal documents from database", deleted)
