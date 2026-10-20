"""DocumentSyncService -- two-phase DB sync for CLI ingestion.

Phase 1: resolve/create document rows (single committed transaction).
Phase 2: per document -- versioning + chunks + outbox (one transaction).

Extracted from IngestionService to isolate the Postgres sync concern.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Unpack

from application.act_versioning import VersioningResult
from application.documents.delete import delete_document_in_uow
from application.indexing import enrich_chunks_metadata, process_chunks, tag_chunks
from application.ingestion.profiles import get_domain_profile
from domain.identity.roles import UserKind
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.document import Document as DocEntity
from domain.knowledge.enums import DocDomain, DocumentVisibility
from domain.shared.exceptions import EntityNotFound, ValidationError


if TYPE_CHECKING:
    from application.act_versioning import ActVersioningService, VersioningPlan
    from application.ingestion.dto import RegistryRecord, SyncOneDocumentOptions
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.protocol import DocumentProfile
    from domain.ingestion.profiles.registry import DomainProfileRegistry

log = logging.getLogger("default")


class DocumentSyncService:
    """Syncs documents and chunks to Postgres, enqueues outbox for Qdrant."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        act_versioning_service: ActVersioningService | None = None,
        domain_registry: DomainProfileRegistry | None = None,
        domain_settings: DomainSettingsPort | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._act_versioning_service = act_versioning_service
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings

    def get_profile(self, domain: str) -> DocumentProfile | None:
        return get_domain_profile(self._domain_registry, domain)

    async def sync_documents_to_db(
        self,
        registry: dict[str, RegistryRecord],
        source_chars: dict[str, int],
        chunks: list[RawDocument],
        full_text_by_source: dict[str, str] | None = None,
        *,
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        client_id: int | None = None,
    ) -> None:
        """Sync documents and chunks to Postgres, enqueue outbox for Qdrant.

        Two phases: resolve document rows, then persist each document atomically:
          Phase 1: resolve/create document rows (committed).
          Phase 2: per document -- versioning + chunks + outbox (one transaction).
        """
        del source_chars  # Chunk persistence computes document statistics.
        full_text_by_source = full_text_by_source or {}
        visibility = DocumentVisibility.validate(visibility)

        owner_id = client_id if visibility == DocumentVisibility.CLIENT_PRIVATE else None
        group_id = group_id if visibility == DocumentVisibility.INTERNAL_GROUP else None

        # Group chunks by source for per-document processing
        chunks_by_source: dict[str, list[RawDocument]] = {}

        for c in chunks:
            src = c.metadata.get("source", "")
            chunks_by_source.setdefault(src, []).append(c)

        documents_to_sync = {
            fname: info for fname, info in registry.items() if chunks_by_source.get(info.get("source", ""))
        }

        # Finish version preparation before creating or changing any documents.
        prepared = {}

        for key, info in documents_to_sync.items():
            source = info["source"]
            domain = chunks_by_source[source][0].metadata.get("doc_domain", DocDomain.GENERAL.value)
            prepared[key] = await self.prepare_versioning(key, domain, full_text_by_source.get(source, ""))

        # --- Phase 1: document rows (single committed transaction) ---
        doc_ids = await self.ensure_document_rows(
            documents_to_sync, visibility=visibility, group_id=group_id, owner_id=owner_id
        )

        # --- Phase 2: versioning + chunks + outbox (per document) ---
        for fname, info in documents_to_sync.items():
            ids = doc_ids.get(fname)

            if not ids:
                continue

            src = info.get("source", "")
            file_chunks = chunks_by_source.get(src, [])

            if not file_chunks:
                continue

            for doc_id in ids:
                await self.sync_one_document(
                    fname=info.get("filename", PurePosixPath(fname).name),
                    doc_id=doc_id,
                    src=src,
                    file_chunks=file_chunks,
                    prepared=prepared[fname],
                    full_text=full_text_by_source.get(src, ""),
                    visibility=visibility,
                    owner_id=owner_id,
                    group_id=group_id,
                )

        log.info("Synced %d documents to database via outbox", len(documents_to_sync))

    async def ensure_document_rows(
        self,
        registry: dict[str, RegistryRecord],
        visibility: DocumentVisibility = DocumentVisibility.INTERNAL_PUBLIC,
        group_id: int | None = None,
        owner_id: int | None = None,
    ) -> dict[str, list[int]]:
        """Resolve sources independently of mutable assignments."""
        doc_ids: dict[str, list[int]] = {}
        async with self._uow_factory.create(master=True) as uow:
            existing_docs = await uow.documents.find_ingestion_documents(list(registry))

            for doc in existing_docs:
                if doc.id is not None and doc.ingestion_key is not None:
                    doc_ids.setdefault(doc.ingestion_key, []).append(doc.id)

            missing = [key for key in registry if key not in doc_ids]

            if missing:
                await self.validate_client_owner(uow, visibility, owner_id)

                if visibility == DocumentVisibility.INTERNAL_GROUP and group_id is None:
                    msg = "group_id required for internal_group ingestion"
                    raise ValidationError(msg)

            for key in missing:
                doc = DocEntity(
                    filename=registry[key].get("filename", PurePosixPath(key).name),
                    ingestion_key=key,
                    visibility=visibility,
                    owner_id=owner_id,
                    group_id=group_id,
                )
                saved = await uow.documents.save(doc)

                if saved.id is None:
                    msg = f"Document {key} has None id after save"
                    raise RuntimeError(msg)

                doc_ids[key] = [saved.id]
        return doc_ids

    async def sync_one_document(
        self,
        fname: str,
        doc_id: int,
        src: str,
        file_chunks: list[RawDocument],
        *,
        prepared: tuple[VersioningResult, VersioningPlan | None, DocumentProfile | None] | None = None,
        **options: Unpack[SyncOneDocumentOptions],
    ) -> None:
        """Persist version state, chunks and outbox together for one document."""
        full_text = options["full_text"]
        first_chunk = file_chunks[0]
        chunk_domain = first_chunk.metadata.get("doc_domain", DocDomain.GENERAL.value)

        versioning, versioning_plan, profile = prepared or await self.prepare_versioning(
            fname, chunk_domain, full_text
        )
        act_versioning_service = self._act_versioning_service

        async with self._uow_factory.create(master=True) as uow:
            document = await uow.documents.get_by_id(doc_id, for_update=True)

            if document is None:
                entity_name = "Document"
                raise EntityNotFound(entity_name, doc_id)

            visibility, owner_id, group_id = document.visibility, document.owner_id, document.group_id
            # Copy per target: one source can have several different ACL assignments.
            raw_chunks = [
                RawDocument(page_content=c.page_content, metadata={**c.metadata, "source": src})
                for c in file_chunks
            ]
            tag_chunks(
                raw_chunks,
                visibility=visibility,
                owner_id=owner_id,
                group_id=group_id,
                client_id=owner_id if visibility == DocumentVisibility.CLIENT_PRIVATE else None,
            )

            if versioning_plan is not None:
                if act_versioning_service is None or profile is None:
                    msg = "Versioning plan exists without an act versioning service"
                    raise RuntimeError(msg)

                versioning = await act_versioning_service.apply_versioning_plan(
                    uow, profile, doc_id, versioning_plan
                )

            enrich_chunks_metadata(
                raw_chunks,
                doc_id,
                visibility.value,
                owner_id,
                group_id,
                doc_domain=chunk_domain,
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

            source_key = src.split("/", 3)[3] if src.startswith("s3://") else src
            await uow.documents.set_source_path(doc_id, source_key)
            await process_chunks(
                uow_factory=self._uow_factory,
                document_id=doc_id,
                filename=fname,
                chunks=raw_chunks,
                visibility=visibility.value,
                owner_id=owner_id,
                group_id=group_id,
                doc_domain=chunk_domain,
                _existing_uow=uow,
            )

        if act_versioning_service is not None:
            if versioning.act_id is not None:
                await act_versioning_service.invalidate_act_answers(versioning.act_id)
            else:
                await act_versioning_service.invalidate_document_answers(doc_id)

    async def delete_missing_sources(self, prefix: str, current_keys: set[str]) -> None:
        """Prune only ingestion sources; SQL, timeline and outbox commit together."""
        async with self._uow_factory.create(master=True) as uow:
            documents = await uow.documents.list_ingestion_documents(prefix)

            for document in documents:
                if document.ingestion_key in current_keys or document.id is None:
                    continue

                locked = await uow.documents.get_by_id(document.id, for_update=True)

                if locked is not None and locked.ingestion_key == document.ingestion_key:
                    await delete_document_in_uow(uow, document.id)

            entries = await uow.ingestion_registry.list_all()

            for key in entries:
                if key.startswith(prefix) and key not in current_keys:
                    await uow.ingestion_registry.delete(key)

    @staticmethod
    async def validate_client_owner(
        uow: UnitOfWork, visibility: DocumentVisibility, owner_id: int | None
    ) -> None:
        if visibility == DocumentVisibility.CLIENT_PRIVATE:
            if owner_id is None:
                msg = "client_id required for client_private ingestion"
                raise ValidationError(msg)

            client_user = await uow.users.get_by_id(owner_id)

            if client_user is None or client_user.kind != UserKind.CLIENT:
                msg = "client_id must be a user with kind='client'"
                raise ValidationError(msg)

    async def prepare_versioning(
        self, fname: str, chunk_domain: str, full_text: str
    ) -> tuple[VersioningResult, VersioningPlan | None, DocumentProfile | None]:
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
            versioning_plan = await act_versioning_service.prepare_document_versioning(profile, full_text)

            if versioning_plan is not None:
                versioning = VersioningResult(
                    versioning_plan.domain_metadata,
                    None,
                    None,
                    versioning_plan.effective_from,
                    versioning_plan.warning,
                )

            if versioning.warning:
                log.warning("Versioning warning for %s: %s", fname, versioning.warning)

        return versioning, versioning_plan, profile
