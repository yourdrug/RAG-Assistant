"""Document Processor — application service for processing uploaded documents.

Uses UoWFactory to manage its own transaction. No db/session parameters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_parser import DocumentParser, DocumentSplitter
from infrastructure.ml.ingestion import extract_date_from_filename
from infrastructure.ml.metrics import (
    INGEST_CHUNKS_TOTAL,
    INGEST_DOCUMENT_DURATION,
    INGEST_DOCUMENTS_TOTAL,
    INGEST_PDF_BAD_RATIO,
    INGEST_PDF_PAGES_TOTAL,
)
from infrastructure.storage import FileStorage
from infrastructure.uow_factory import UnitOfWorkFactory

log = logging.getLogger("default")


class DocumentProcessor:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_store_repo: VectorStoreRepository,
        file_storage: FileStorage,
        document_parser: DocumentParser,
        document_splitter: DocumentSplitter,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_store = vector_store_repo
        self._file_storage = file_storage
        self._parser = document_parser
        self._splitter = document_splitter

    async def process(
        self,
        document_id: int,
        storage_key: str,
        original_filename: str,
        visibility: str,
        owner_id: int | None,
        group_id: int | None,
        replace_id: int | None,
    ) -> None:
        t_start = time.monotonic()
        temp_path: Path | None = None
        status = "failed"
        try:
            async with self._uow_factory.create() as uow:
                await uow.documents.update_status(document_id, "processing")

                temp_path = self._file_storage.download_to_temp(storage_key)
                docs = self._parser.parse(temp_path)

                if not docs:
                    raise RuntimeError(
                        "Текст не извлечён — документ похож на скан, " "и OCR не смог распознать содержимое."
                    )

                warning_message = None
                if Path(original_filename).suffix.lower() == ".pdf":
                    from infrastructure.ml.pdf_diag import assess_pdf_extraction_quality

                    quality = assess_pdf_extraction_quality(temp_path, docs)
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

                    INGEST_PDF_PAGES_TOTAL.labels(quality="ok").inc(quality.n_ok)
                    INGEST_PDF_PAGES_TOTAL.labels(quality="missing").inc(quality.n_missing)
                    INGEST_PDF_PAGES_TOTAL.labels(quality="garbled").inc(quality.n_garbled)
                    INGEST_PDF_BAD_RATIO.observe(quality.bad_ratio)

                doc_date = extract_date_from_filename(original_filename)

                for doc in docs:
                    doc.metadata["source"] = original_filename
                    if doc_date:
                        doc.metadata["doc_date"] = doc_date

                chunks = self._splitter.split(docs)
                for chunk in chunks:
                    chunk.metadata.update(
                        {
                            "document_id": document_id,
                            "visibility": visibility,
                            "owner_id": owner_id,
                            "group_id": group_id,
                        }
                    )
                    section = chunk.metadata.get("section")
                    if section:
                        chunk.page_content = f"[Раздел: {section}]\n{chunk.page_content}"

                from domain.entities.chunk import Chunk

                domain_chunks = [Chunk(content=c.page_content, metadata=c.metadata) for c in chunks]

                vector_size = len(self._vector_store.generate_embeddings("test"))
                self._vector_store.ensure_collection(vector_size, reset=False)
                self._vector_store.upload_documents(domain_chunks)

                if replace_id is not None:
                    self._vector_store.delete_by_document_id(replace_id)
                    old = await uow.documents.get_by_id(replace_id)
                    if old and old.source_path:
                        self._file_storage.delete_file(old.source_path)
                    await uow.documents.delete(replace_id)

                total_chars = sum(len(d.page_content) for d in docs)
                await uow.documents.update_status(
                    document_id, "done", chunks=len(chunks), chars=total_chars, warning=warning_message
                )

            status = "done"
            INGEST_CHUNKS_TOTAL.inc(len(chunks))

        except Exception as e:
            log.exception("Document processing failed for doc %d: %s", document_id, e)
            try:
                async with self._uow_factory.create() as uow:
                    await uow.documents.update_status(document_id, "failed", error=str(e))
            except Exception:
                log.exception("Failed to mark document as failed")
        finally:
            INGEST_DOCUMENTS_TOTAL.labels(status=status).inc()
            INGEST_DOCUMENT_DURATION.labels(status=status).observe(time.monotonic() - t_start)
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
