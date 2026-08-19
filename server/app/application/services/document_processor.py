"""Application service for processing uploaded documents end-to-end.

Orchestrates the pipeline: download from storage, parse, split, generate
embeddings, upload to vector store, and persist chunk metadata to Postgres.
Reports quality warnings for low-fidelity PDF extractions and records
Prometheus metrics for each processing stage.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from domain.entities.chunk import Chunk
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_domain_classifier import classify_document_domain
from domain.services.document_parser import DocumentParser, DocumentSplitter
from domain.value_objects.document_status import DocumentStatus

from application.ports.document_processing import (
    ContentExtractorPort,
    MetricsCollectorPort,
    PDFQualityAssessorPort,
)
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory

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

    def _assess_pdf_quality_for_docs(
        self, temp_path: Path, original_filename: str, document_id: int, docs: list
    ) -> tuple[object | None, str | None]:
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
    def _enrich_chunks(
        raw_chunks,
        doc_domain: str,
        document_id: int,
        visibility: str,
        owner_id: int | None,
        group_id: int | None,
    ) -> None:
        for rc in raw_chunks:
            rc.metadata.update(
                {
                    "document_id": document_id,
                    "visibility": visibility,
                    "owner_id": owner_id,
                    "group_id": group_id,
                    "doc_domain": doc_domain,
                }
            )

    async def _upload_to_vector_store(self, domain_chunks: list) -> None:
        vector_size = len(await self._vector_store.generate_embeddings("test"))
        await self._vector_store.ensure_collection(vector_size, reset=False)
        await self._vector_store.upload_documents(domain_chunks)

    async def _replace_existing_document(self, uow, replace_id: int) -> None:
        await self._vector_store.delete_by_document_id(replace_id)
        old = await uow.documents.get_by_id(replace_id)
        if old and old.source_path:
            self._file_storage.delete_file(old.source_path)
        await uow.documents.delete(replace_id)

    async def _update_document_status(
        self,
        uow,
        document_id: int,
        raw_chunks: list,
        docs: list,
        warning_message: str | None,
        quality,
    ) -> None:
        total_chars = sum(len(d.page_content) for d in docs)
        await uow.documents.update_status(
            document_id,
            DocumentStatus.DONE.value,
            chunks=len(raw_chunks),
            chars=total_chars,
            warning=warning_message,
            quality_score=quality.bad_ratio if quality else None,
        )

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

    @staticmethod
    def _enrich_chunk_with_section(rc, doc_domain: str) -> None:
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
            temp_path = self._file_storage.download_to_temp(storage_key)
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
            self._enrich_chunks(raw_chunks, doc_domain, document_id, visibility, owner_id, group_id)

            for rc in raw_chunks:
                self._enrich_chunk_with_section(rc, doc_domain)

            domain_chunks = [Chunk(content=rc.page_content, metadata=rc.metadata) for rc in raw_chunks]
            await self._upload_to_vector_store(domain_chunks)

            # --- Short transaction: persist chunks + mark DONE ---
            async with self._uow_factory.create(master=True) as uow:
                await uow.documents.set_domain(document_id, doc_domain)

                await uow.chunks.bulk_insert(
                    document_id=document_id,
                    filename=original_filename,
                    visibility=visibility,
                    chunks=[rc.page_content for rc in raw_chunks],
                    owner_id=owner_id,
                    group_id=group_id,
                    doc_domain=doc_domain,
                )

                if replace_id is not None:
                    await self._replace_existing_document(uow, replace_id)

                await self._update_document_status(
                    uow,
                    document_id,
                    raw_chunks,
                    docs,
                    warning_message,
                    quality,
                )

            status = DocumentStatus.DONE.value

        except Exception as e:
            await self._handle_processing_failure(document_id, e)
        finally:
            self._finalize_processing(status, t_start, temp_path, raw_chunks)
