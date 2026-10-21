"""Coordinate uploaded-document processing and its resource lifecycle.

Content preparation, classification, replacement and versioning live in
sibling modules. Persistence owns the transaction that commits versions,
chunks and vector outbox operations together; storage cleanup follows commit.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from application.documents.classification import classify_processing_document
from application.documents.content import DocumentContentProcessor
from application.documents.dto import PersistDocumentCommand
from application.documents.persistence import persist_document_result
from application.documents.processing_context import ProcessingContext
from application.documents.versioning import prepare_processing_versioning
from domain.knowledge.enums import DocumentStatus


if TYPE_CHECKING:
    from application.act_versioning import ActVersioningService
    from application.documents.dto import (
        DocumentProcessingDependencies,
        PreparedDocumentVersioning,
        ProcessDocumentCommand,
    )
    from application.ports.file_storage import FileStorage
    from application.uow import UnitOfWorkFactory
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from domain.knowledge.document import Document

log = logging.getLogger("default")


class DocumentProcessor:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        file_storage: FileStorage,
        dependencies: DocumentProcessingDependencies,
        *,
        domain_registry: DomainProfileRegistry | None = None,
        domain_settings: DomainSettingsPort | None = None,
        act_versioning_service: ActVersioningService | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._file_storage = file_storage
        self._metrics = dependencies.metrics
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings
        self._act_versioning_service = act_versioning_service
        self._content = DocumentContentProcessor(
            dependencies.document_parser,
            dependencies.document_splitter,
            dependencies.content_extractor,
            dependencies.pdf_quality_assessor,
            text_assessor=dependencies.text_quality_assessor,
            metrics=dependencies.metrics,
        )

    async def get_document(self, document_id: int) -> Document | None:
        """Fetch a document by id, returning None if not found."""
        async with self._uow_factory.create() as uow:
            return await uow.documents.get_by_id(document_id)

    async def handle_processing_failure(self, document_id: int, e: Exception) -> None:
        log.error("Document processing failed for doc %d: %s", document_id, e, exc_info=e)
        try:
            async with self._uow_factory.create(master=True) as uow:
                await uow.documents.update_status(
                    document_id, status=DocumentStatus.FAILED.value, error=str(e)
                )
        except Exception:
            log.exception("Failed to mark document as failed")

    def finalize_processing(self, ctx: ProcessingContext, t_start: float) -> None:
        self._metrics.inc_documents(ctx.status)
        self._metrics.observe_duration(ctx.status, time.monotonic() - t_start)

        if ctx.temp_path is not None:
            ctx.temp_path.unlink(missing_ok=True)

        if ctx.raw_chunks:
            self._metrics.inc_chunks(len(ctx.raw_chunks))

    async def process(self, command: ProcessDocumentCommand) -> None:
        document_id = command.document_id
        ctx = ProcessingContext(
            document_id=document_id,
            replace_id=command.replace_id,
            doc_domain=command.doc_domain,
        )
        t_start = time.monotonic()
        try:
            async with self._uow_factory.create(master=True) as uow:
                await uow.documents.update_status(document_id, status=DocumentStatus.PROCESSING.value)

            ctx.temp_path = await self._file_storage.download_to_temp(command.storage_key)
            content = await self._content.parse(ctx.temp_path, command.original_filename, document_id)

            full_text = "\n".join(d.page_content for d in content.documents)

            classification = classify_processing_document(
                document_id,
                full_text,
                command.doc_domain,
                domain_registry=self._domain_registry,
                domain_settings=self._domain_settings,
                metrics=self._metrics,
            )
            ctx.doc_domain = classification.domain_key
            ctx.warnings.extend(warning for warning in (content.warning, classification.warning) if warning)

            raw_chunks = await self._content.split(
                content.documents, command.original_filename, classification.domain_key
            )
            ctx.raw_chunks = raw_chunks

            prepared_versioning: PreparedDocumentVersioning | None = await prepare_processing_versioning(
                classification.domain_key,
                full_text,
                domain_registry=self._domain_registry,
                act_versioning_service=self._act_versioning_service,
            )

            if prepared_versioning is not None and prepared_versioning.plan.warning:
                ctx.warnings.append(prepared_versioning.plan.warning)

            if await self.get_document(document_id) is None:
                log.info("Document %d was deleted during processing — aborting", document_id)
                return

            persisted_versioning = await persist_document_result(
                self._uow_factory,
                PersistDocumentCommand(
                    document_id=document_id,
                    original_filename=command.original_filename,
                    raw_chunks=raw_chunks,
                    visibility=command.visibility,
                    owner_id=command.owner_id,
                    group_id=command.group_id,
                    doc_domain=classification.domain_key,
                    replace_id=command.replace_id,
                    warning_message=ctx.warning_message,
                    quality=content.quality,
                    storage_deletes=ctx.storage_deletes,
                    versioning=prepared_versioning,
                ),
                domain_registry=self._domain_registry,
                act_versioning_service=self._act_versioning_service,
            )

            if persisted_versioning is None:
                return

            ctx.status = DocumentStatus.INDEXING

        except Exception as e:
            await self.handle_processing_failure(document_id, e)
            raise
        finally:
            self.finalize_processing(ctx, t_start)
            # Runs on EVERY exit path (incl. early abort returns): otherwise a
            # replaced object listed in storage_deletes is orphaned in S3.
            await self.cleanup_storage_objects(ctx.storage_deletes)

    async def cleanup_storage_objects(self, keys: list[str]) -> None:
        for old_key in keys:
            try:
                await self._file_storage.delete_file(old_key)
            except Exception:
                log.exception("Failed to delete replaced object %s from storage — orphaned", old_key)
