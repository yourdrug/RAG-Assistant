"""Application service for processing uploaded documents end-to-end.

Orchestrates the pipeline: download from storage, parse, split, persist
chunk metadata to Postgres, and enqueue vector-store operations via the
Transactional Outbox pattern.  The outbox dispatcher applies changes to
Qdrant asynchronously after the Postgres transaction commits.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from application.services.document_pipeline import enrich_chunks_metadata, process_chunks
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_domain_classifier import classify_document_domain
from domain.services.document_parser import DocumentParser, DocumentSplitter
from domain.value_objects.document_status import DocumentStatus

from application.ports.document_processing import (
    ContentExtractorPort,
    MetricsCollectorPort,
    PDFQualityAssessorPort,
    PDFQualityReport,
)
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory

if TYPE_CHECKING:
    from infrastructure.ml.client_registry import MLClientRegistry

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
        metrics: MetricsCollectorPort,
        domain_marker_threshold: float = 1.0,
        ml_registry: MLClientRegistry | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store_repo
        self._file_storage = file_storage
        self._parser = document_parser
        self._splitter = document_splitter
        self._extractor = content_extractor
        self._pdf_assessor = pdf_quality_assessor
        self._metrics = metrics
        self._domain_marker_threshold = domain_marker_threshold
        self._ml_registry = ml_registry

    def _assess_pdf_quality_for_docs(
        self, temp_path: Path, original_filename: str, document_id: int, docs: list
    ) -> tuple[PDFQualityReport | None, str | None]:
        if Path(original_filename).suffix.lower() != ".pdf":
            return None, None
        quality = self._pdf_assessor.assess(temp_path, docs)
        warning_message = None
        if quality.is_low_quality:
            warning_message = (
                f"Низкое качество распознавания: {quality.n_missing} стр. без текста, "
                f"{quality.n_garbled} стр. с мусорным текстом из {quality.total_pages}. "
                "Рекомендуется проверить документ (task pdf:diag) и переиндексировать "
                "после конвертации или ручной вычитки."
            )
            log.warning(
                "Low-quality extraction for doc %d (%s): bad_ratio=%.2f",
                document_id,
                original_filename,
                quality.bad_ratio,
            )
        self._metrics.observe_pdf_pages("ok", quality.n_ok)
        self._metrics.observe_pdf_pages("missing", quality.n_missing)
        self._metrics.observe_pdf_pages("garbled", quality.n_garbled)
        self._metrics.observe_pdf_bad_ratio(quality.bad_ratio)
        return quality, warning_message

    @staticmethod
    def _enrich_chunk_with_section(rc: Any, doc_domain: str) -> None:
        section = rc.metadata.get("section")
        if section:
            rc.page_content = f"[Раздел: {section}]\n{rc.page_content}"

    @staticmethod
    def _attach_metadata_to_docs(docs: list, original_filename: str, extractor) -> None:
        doc_date = extractor.extract_date_from_filename(original_filename)
        for doc in docs:
            doc.metadata["source"] = original_filename
            if doc_date:
                doc.metadata["doc_date"] = doc_date

    async def _handle_processing_failure(self, document_id: int, e: Exception) -> None:
        log.exception("Document processing failed for doc %d: %s", document_id, e)
        try:
            async with self._uow_factory.create(master=True) as uow:
                await uow.documents.update_status(document_id, DocumentStatus.FAILED.value, error=str(e))
        except Exception:
            log.exception("Failed to mark document as failed")

    def _finalize_processing(
        self,
        status: str,
        t_start: float,
        temp_path: Path | None,
        raw_chunks: list | None,
    ) -> None:
        self._metrics.inc_documents(status)
        self._metrics.observe_duration(status, time.monotonic() - t_start)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if raw_chunks:
            self._metrics.inc_chunks(len(raw_chunks))

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
        t_start = time.monotonic()
        temp_path: Path | None = None
        status = DocumentStatus.FAILED.value
        raw_chunks = None
        try:
            # --- Short transaction: mark as PROCESSING ---
            async with self._uow_factory.create(master=True) as uow:
                await uow.documents.update_status(document_id, DocumentStatus.PROCESSING.value)

            # --- Heavy I/O outside transaction ---
            temp_path = await self._file_storage.download_to_temp(storage_key)
            docs = self._parser.parse(temp_path)

            if not docs:
                raise RuntimeError(
                    "Текст не извлечён — документ похож на скан, и OCR не смог распознать содержимое."
                )

            quality, warning_message = self._assess_pdf_quality_for_docs(
                temp_path,
                original_filename,
                document_id,
                docs,
            )

            if doc_domain is None:
                full_text = "\n".join(d.page_content for d in docs)
                doc_domain = classify_document_domain(full_text, threshold=self._domain_marker_threshold)
                log.info("Auto-detected doc_domain=%s for doc %d", doc_domain, document_id)

            self._attach_metadata_to_docs(docs, original_filename, self._extractor)

            raw_chunks = self._splitter.split(docs, domain=doc_domain)

            # API-specific: section enrichment
            for rc in raw_chunks:
                self._enrich_chunk_with_section(rc, doc_domain)

            # --- Shared pipeline: Postgres + outbox ---
            async with self._uow_factory.create(master=True) as uow:
                await uow.documents.set_domain(document_id, doc_domain)

                # Enrich metadata
                enrich_chunks_metadata(
                    raw_chunks,
                    document_id,
                    visibility,
                    owner_id,
                    group_id,
                    doc_domain,
                )

                # Pipeline: bulk_insert → outbox → indexing status
                await process_chunks(
                    uow_factory=self._uow_factory,
                    document_id=document_id,
                    filename=original_filename,
                    chunks=raw_chunks,
                    visibility=visibility,
                    owner_id=owner_id,
                    group_id=group_id,
                    doc_domain=doc_domain,
                    set_indexing=True,
                )

                # Handle document replacement (API-specific)
                if replace_id is not None:
                    await uow.vector_outbox.enqueue(
                        VectorOutboxEntry(
                            operation=OutboxOperation.DELETE_BY_DOCUMENT,
                            aggregate_type="document",
                            aggregate_id=replace_id,
                            payload={"document_id": replace_id},
                        )
                    )
                    old = await uow.documents.get_by_id(replace_id)
                    if old and old.source_path:
                        self._file_storage.delete_file(old.source_path)
                    await uow.documents.delete(replace_id)

                # Update stats with quality warning
                if warning_message:
                    await uow.documents.update_status(
                        document_id,
                        DocumentStatus.INDEXING.value,
                        warning=warning_message,
                        quality_score=quality.bad_ratio if quality else None,
                    )

            status = DocumentStatus.INDEXING.value

        except Exception as e:
            await self._handle_processing_failure(document_id, e)
        finally:
            self._finalize_processing(status, t_start, temp_path, raw_chunks)
