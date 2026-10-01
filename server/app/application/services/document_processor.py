"""Coordinate uploaded-document processing and its resource lifecycle.

Content preparation, classification, replacement and versioning live in
sibling modules. Persistence owns the transaction that commits versions,
chunks and vector outbox operations together; storage cleanup follows commit.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from application.dto.document_processing_context import ProcessingContext
from application.ports.document_processing import (
    ContentExtractorPort,
    MetricsCollectorPort,
    PDFQualityAssessorPort,
    TextQualityAssessorPort,
)
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_conflict_resolver import resolve_processing_conflict
from application.services.document_content import DocumentContentProcessor
from application.services.document_persistence import persist_document_result
from application.services.document_versioning import prepare_processing_versioning
from application.services.domain_classification import classify_processing_document
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_parser import DocumentParser, DocumentSplitter
from domain.value_objects.document_status import DocumentStatus

if TYPE_CHECKING:
    from application.services.act_versioning_service import ActVersioningService
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.domain_profile.settings_port import DomainSettingsPort

log = logging.getLogger("default")


class DocumentProcessor:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
        document_parser: DocumentParser,
        document_splitter: DocumentSplitter,
        content_extractor: ContentExtractorPort,
        pdf_quality_assessor: PDFQualityAssessorPort,
        text_quality_assessor: TextQualityAssessorPort,
        metrics: MetricsCollectorPort,
        domain_marker_threshold: float = 1.0,
        domain_registry: DomainProfileRegistry | None = None,
        domain_settings: "DomainSettingsPort | None" = None,
        act_versioning_service: "ActVersioningService | None" = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._metrics = metrics
        self._domain_marker_threshold = domain_marker_threshold
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings
        self._act_versioning_service = act_versioning_service
        self._content = DocumentContentProcessor(
            document_parser,
            document_splitter,
            content_extractor,
            pdf_quality_assessor,
            text_quality_assessor,
            metrics,
        )

    async def _get_document(self, document_id: int):
        """Fetch a document by id, returning None if not found."""
        async with self._uow_factory.create() as uow:
            return await uow.documents.get_by_id(document_id)

    async def _handle_processing_failure(self, document_id: int, e: Exception) -> None:
        log.exception("Document processing failed for doc %d: %s", document_id, e)
        try:
            async with self._uow_factory.create(master=True) as uow:
                await uow.documents.update_status(document_id, DocumentStatus.FAILED.value, error=str(e))
        except Exception:
            log.exception("Failed to mark document as failed")

    def _finalize_processing(self, ctx: ProcessingContext, t_start: float) -> None:
        self._metrics.inc_documents(ctx.status)
        self._metrics.observe_duration(ctx.status, time.monotonic() - t_start)
        if ctx.temp_path is not None:
            ctx.temp_path.unlink(missing_ok=True)
        if ctx.raw_chunks:
            self._metrics.inc_chunks(len(ctx.raw_chunks))

    async def process(
        self,
        document_id: int,
        storage_key: str,
        original_filename: str,
        visibility: str,
        owner_id: int | None,
        group_id: int | None,
        replace_id: int | None,
        doc_domain: str | None = None,
    ) -> None:
        ctx = ProcessingContext(
            document_id=document_id,
            storage_key=storage_key,
            original_filename=original_filename,
            visibility=visibility,
            owner_id=owner_id,
            group_id=group_id,
            replace_id=replace_id,
            doc_domain=doc_domain,
        )
        t_start = time.monotonic()
        try:
            async with self._uow_factory.create(master=True) as uow:
                await uow.documents.update_status(document_id, DocumentStatus.PROCESSING.value)

            ctx.temp_path = await self._file_storage.download_to_temp(storage_key)
            await self._content.parse(ctx)

            full_text = "\n".join(d.page_content for d in ctx.docs)

            classify_processing_document(
                ctx,
                full_text,
                domain_registry=self._domain_registry,
                domain_settings=self._domain_settings,
                fallback_threshold=self._domain_marker_threshold,
                metrics=self._metrics,
            )
            await resolve_processing_conflict(self._uow_factory, ctx, self._domain_registry)
            doc_domain = ctx.doc_domain
            if doc_domain is None:
                raise RuntimeError(f"domain classification produced no domain for doc {document_id}")

            raw_chunks = await self._content.split(ctx)

            await prepare_processing_versioning(
                ctx,
                full_text,
                domain_registry=self._domain_registry,
                act_versioning_service=self._act_versioning_service,
            )

            current_doc = await self._get_document(document_id)
            if current_doc is None:
                log.info("Document %d was deleted during processing — aborting", document_id)
                return

            persisted_versioning = await persist_document_result(
                self._uow_factory,
                document_id=document_id,
                original_filename=original_filename,
                raw_chunks=raw_chunks,
                visibility=visibility,
                owner_id=owner_id,
                group_id=group_id,
                doc_domain=doc_domain,
                replace_id=replace_id,
                warning_message=ctx.warning_message,
                quality=ctx.quality,
                storage_deletes=ctx.storage_deletes,
                domain_registry=self._domain_registry,
                versioning=ctx.versioning,
                versioning_plan=ctx.versioning_plan,
                versioning_profile=ctx.versioning_profile,
                act_versioning_service=self._act_versioning_service,
            )
            if persisted_versioning is None:
                return
            ctx.versioning = persisted_versioning

            ctx.status = DocumentStatus.INDEXING.value

        except Exception as e:
            await self._handle_processing_failure(document_id, e)
            raise
        finally:
            self._finalize_processing(ctx, t_start)
            # Runs on EVERY exit path (incl. early abort returns): otherwise a
            # replaced object listed in storage_deletes is orphaned in S3.
            await self._cleanup_storage_objects(ctx.storage_deletes)

    async def _cleanup_storage_objects(self, keys: list[str]) -> None:
        for old_key in keys:
            try:
                await self._file_storage.delete_file(old_key)
            except Exception:
                log.warning("Failed to delete replaced object %s from storage — orphaned", old_key)
