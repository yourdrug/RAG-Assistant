"""DocumentSyncService -- two-phase DB sync for CLI ingestion.

Phase 1: resolve/create document rows (single committed transaction).
Phase 2: per document -- versioning (own tx) + chunks + outbox.

Extracted from IngestionService to isolate the Postgres sync concern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.dto.versioning_dto import VersioningResult
from application.services.document_pipeline import enrich_chunks_metadata, process_chunks
from domain.entities.document import Document as DocEntity
from domain.entities.raw_document import RawDocument
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.visibility import DocumentVisibility

if TYPE_CHECKING:
    from domain.domain_profile.settings_port import DomainSettingsPort
    from application.ports.unit_of_work_factory import UnitOfWorkFactory
    from application.services.act_versioning_service import ActVersioningService
    from domain.domain_profile.registry import DomainProfileRegistry

log = logging.getLogger("default")


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

    def _get_profile(self, domain: str):
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

        # Group chunks by source for per-document processing
        chunks_by_source: dict[str, list] = {}
        for c in chunks:
            src = c.metadata.get("source", "")
            chunks_by_source.setdefault(src, []).append(c)

        # --- Phase 1: document rows (single committed transaction) ---
        doc_ids = await self._ensure_document_rows(
            registry, visibility=visibility, group_id=group_id, client_id=client_id
        )

        # --- Phase 2: versioning + chunks + outbox (per document) ---
        for fname, info in registry.items():
            doc_id = doc_ids.get(fname)
            if doc_id is None:
                continue
            src = info.get("source", "")
            file_chunks = chunks_by_source.get(src, [])
            if not file_chunks:
                continue
            await self._sync_one_document(
                fname=fname,
                doc_id=doc_id,
                src=src,
                file_chunks=file_chunks,
                full_text=full_text_by_source.get(src, ""),
            )

        log.info("Synced %d documents to database via outbox", len(registry))

    async def _ensure_document_rows(
        self,
        registry: dict,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> dict[str, int]:
        """Phase 1: get-or-create document rows for all registry filenames."""
        doc_ids: dict[str, int] = {}
        async with self._uow_factory.create(master=True) as uow:
            owner_id = None  # CLI ingest has no user context

            for fname in registry:
                existing = await uow.documents.find_active_slot(
                    owner_id=owner_id,
                    filename=fname,
                    group_id=group_id,
                )
                if existing:
                    if existing.id is None:
                        raise RuntimeError(f"Document {fname} has None id after find_active_slot")
                    doc_ids[fname] = existing.id
                else:
                    doc = DocEntity(
                        filename=fname,
                        visibility=visibility,
                        owner_id=owner_id,
                        group_id=group_id,
                    )
                    saved = await uow.documents.save(doc)
                    if saved.id is None:
                        raise RuntimeError(f"Document {fname} has None id after save")
                    doc_ids[fname] = saved.id
        return doc_ids

    async def _sync_one_document(
        self,
        fname: str,
        doc_id: int,
        src: str,
        file_chunks: list,
        full_text: str,
    ) -> None:
        """Persist version state, chunks and outbox together for one document."""
        first_chunk = file_chunks[0]
        vis = first_chunk.metadata.get("visibility", "internal_public")
        owner = first_chunk.metadata.get("owner_id")
        group = first_chunk.metadata.get("group_id")
        chunk_domain = first_chunk.metadata.get("doc_domain", DocDomain.GENERAL.value)

        versioning = VersioningResult(
            domain_metadata=None,
            act_version_id=None,
            act_id=None,
            effective_from=None,
            warning=None,
        )
        profile = self._get_profile(chunk_domain)
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

        raw_chunks = [
            RawDocument(page_content=c.page_content, metadata=dict(c.metadata)) for c in file_chunks
        ]

        async with self._uow_factory.create(master=True) as uow:
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
                vis,
                owner,
                group,
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
                visibility=vis,
                owner_id=owner,
                group_id=group,
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
        if act_versioning_service is not None:
            if versioning.act_id is not None:
                await act_versioning_service.invalidate_act_answers(versioning.act_id)
            else:
                await act_versioning_service.invalidate_document_answers(doc_id)

    async def delete_internal_documents(self) -> None:
        """Delete all internal documents (owner_id=NULL, non-manual) from DB.

        Called during reset to remove stale document records whose vectors
        have already been deleted from Qdrant.
        """
        async with self._uow_factory.create(master=True) as uow:
            deleted = await uow.documents.delete_internal_documents()
            log.info("Deleted %d internal documents from database", deleted)
