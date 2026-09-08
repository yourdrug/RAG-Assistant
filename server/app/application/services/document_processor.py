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

from application.dto.versioning_dto import VersioningResult
from application.ports.document_processing import (
    ContentExtractorPort,
    MetricsCollectorPort,
    PDFQualityAssessorPort,
    PDFQualityReport,
)
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_pipeline import enrich_chunks_metadata, process_chunks
from domain.entities.vector_outbox_entry import OutboxOperation, VectorOutboxEntry
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_domain_classifier import classify_document_domain
from domain.services.document_parser import DocumentParser, DocumentSplitter
from domain.value_objects.document_status import DocumentStatus

if TYPE_CHECKING:
    from application.ports.domain_settings import DomainSettingsPort
    from infrastructure.domain_profile.registry import DomainProfileRegistry
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
        domain_registry: DomainProfileRegistry | None = None,
        domain_settings: "DomainSettingsPort | None" = None,
        act_versioning_service=None,
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
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings
        self._act_versioning_service = act_versioning_service

    def _assess_quality_for_docs(
        self, temp_path: Path, original_filename: str, document_id: int, docs: list
    ) -> tuple[PDFQualityReport | None, str | None]:
        """Assess document quality and return warning if issues found.

        Supports PDF, DOCX, and RTF formats.
        """
        suffix = Path(original_filename).suffix.lower()
        warning_message = None

        if suffix == ".pdf":
            quality = self._pdf_assessor.assess(temp_path, docs)
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

        # Generic quality check for other formats (DOCX, RTF, TXT, MD)
        from infrastructure.ml.pdf_diag import is_garbled

        total_chars = sum(len(d.page_content) for d in docs)
        garbled_pages = sum(1 for d in docs if is_garbled(d.page_content))
        empty_pages = sum(1 for d in docs if not d.page_content.strip())
        total_pages = len(docs)

        if total_pages == 0:
            return None, None

        bad_ratio = (garbled_pages + empty_pages) / total_pages if total_pages > 0 else 0

        if bad_ratio > 0.3:
            warning_message = (
                f"Низкое качество извлечения текста: "
                f"{garbled_pages} стр. с мусорным текстом, "
                f"{empty_pages} пустых стр. из {total_pages}. "
                f"Рекомендуется проверить документ."
            )
            log.warning(
                "Low-quality extraction for doc %d (%s): garbled=%d, empty=%d, total=%d",
                document_id,
                original_filename,
                garbled_pages,
                empty_pages,
                total_pages,
            )
        elif total_chars < 50 and total_pages > 0:
            warning_message = (
                f"Документ содержит очень мало текста ({total_chars} символов). "
                "Возможно, это скан или повреждённый файл."
            )
            log.warning(
                "Very low text content for doc %d (%s): %d chars",
                document_id,
                original_filename,
                total_chars,
            )

        return None, warning_message

    @staticmethod
    def _enrich_chunk_with_section(rc: Any, doc_domain: str) -> None:
        section = rc.metadata.get("section")
        if section:
            rc.page_content = f"[Раздел: {section}]\n{rc.page_content}"

    def _classify_domain(self, full_text: str, document_id: int) -> tuple[str, str | None]:
        """Registry-based classification with legacy-marksman fallback.

        Returns (domain_key, ambiguous_warning | None). Ambiguous results are
        never swallowed — they surface in documents.warning_message.
        """
        if self._domain_registry is None or self._domain_settings is None:
            log.warning("DomainRegistry unavailable -- falling back to legacy classifier")
            return (
                classify_document_domain(full_text, threshold=self._domain_marker_threshold),
                None,
            )
        result = self._domain_registry.classify(full_text, settings=self._domain_settings)
        self._metrics.observe_domain_classification(result.domain_key, "document", result.confidence)
        if not (result.is_ambiguous or result.confidence < 0.5):
            return result.domain_key, None
        scores_str = ", ".join(
            f"{k}={v:.1f}" for k, v in result.candidate_scores.items() if not k.startswith("__")
        )
        self._metrics.inc_domain_ambiguous(
            ",".join(sorted(k for k in result.candidate_scores if not k.startswith("__")))
            or result.domain_key
        )
        log.warning("Ambiguous classification for doc %d: %s", document_id, scores_str)
        return (
            result.domain_key,
            f"Классификация домена неоднозначна ({scores_str}) — требуется ручная проверка",
        )

    async def _handle_versioning(self, document_id: int, doc_domain: str, full_text: str) -> VersioningResult:
        """Delegate to the unified versioning mechanism (shared with CLI ingestion)."""
        empty = VersioningResult(
            domain_metadata=None,
            act_version_id=None,
            act_id=None,
            effective_from=None,
            warning=None,
        )
        if self._domain_registry is None or self._act_versioning_service is None:
            return empty
        try:
            profile = self._domain_registry.get(doc_domain)
        except KeyError:
            return empty
        return await self._act_versioning_service.process_document_versioning(profile, document_id, full_text)

    async def _get_document(self, document_id: int):
        """Fetch a document by id, returning None if not found."""
        async with self._uow_factory.create() as uow:
            return await uow.documents.get_by_id(document_id)

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

    async def process(  # noqa: C901
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
        storage_deletes: list[str] = []
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

            quality, warning_message = self._assess_quality_for_docs(
                temp_path,
                original_filename,
                document_id,
                docs,
            )

            full_text = "\n".join(d.page_content for d in docs)

            # --- Domain classification: registry-based with fallback ---
            if doc_domain is None:
                doc_domain, ambiguous_warning = self._classify_domain(full_text, document_id)
                if ambiguous_warning:
                    warning_message = (
                        f"{warning_message}\n{ambiguous_warning}" if warning_message else ambiguous_warning
                    )
                log.info("Auto-detected doc_domain=%s for doc %d", doc_domain, document_id)

            # --- Async resolve: if conflict was pending and domain is now known ---
            if replace_id is not None:
                from application.services.document_conflict_resolver import resolve_conflict

                new_doc = await self._get_document(document_id)
                old_doc = await self._get_document(replace_id) if replace_id else None
                if new_doc and old_doc:
                    profile = self._domain_registry.get(doc_domain) if self._domain_registry else None
                    await resolve_conflict(
                        self._uow_factory,
                        new_doc,
                        old_doc,
                        profile,
                        act_versioning_service=self._act_versioning_service,
                    )

            self._attach_metadata_to_docs(docs, original_filename, self._extractor)

            raw_chunks = self._splitter.split(docs, domain=doc_domain)

            # API-specific: section enrichment
            for rc in raw_chunks:
                self._enrich_chunk_with_section(rc, doc_domain)

            # --- Extract references and effective date for versioned domains ---
            versioning = await self._handle_versioning(document_id, doc_domain, full_text)
            domain_metadata = versioning.domain_metadata
            act_version_id = versioning.act_version_id
            act_id = versioning.act_id
            effective_from = versioning.effective_from
            if versioning.warning:
                warning_message = (
                    f"{warning_message}\n{versioning.warning}" if warning_message else versioning.warning
                )

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
                    domain_metadata=domain_metadata,
                    act_version_id=act_version_id,
                    act_id=act_id,
                    effective_from=effective_from,
                )

                # Pipeline: bulk_insert → outbox → indexing status.
                # _existing_uow keeps chunks + outbox + status in THIS
                # transaction: an inner commit here would let the outbox
                # dispatcher race ahead and mark the document DONE before the
                # outer transaction (warning, replacement) commits.
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
                    _existing_uow=uow,
                )

                # Handle document replacement (API-specific)
                if replace_id is not None:
                    await self._handle_replacement(uow, replace_id, doc_domain, storage_deletes)

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

        # Storage mutation AFTER the transaction committed. On failure the
        # DB stays consistent; the orphaned S3 object is harmless garbage.
        await self._cleanup_storage_objects(storage_deletes)

    async def _cleanup_storage_objects(self, keys: list[str]) -> None:
        for old_key in keys:
            try:
                await self._file_storage.delete_file(old_key)
            except Exception:
                log.warning("Failed to delete replaced object %s from storage — orphaned", old_key)

    async def _handle_replacement(
        self, uow, replace_id: int, doc_domain: str, storage_deletes: list[str]
    ) -> None:
        """Delete the replaced document — unless the domain is versioned.

        Versioned domain: the previous edition must stay — its ActVersion was
        already flipped to is_current=False by handle_versioned_upload for the
        new document.
        """
        replace_profile = self._domain_registry.get(doc_domain) if self._domain_registry is not None else None
        if replace_profile is not None and replace_profile.is_versioned:
            log.info("Keeping previous edition doc %d (versioned domain %s)", replace_id, doc_domain)
            return
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
            # the S3 object is deleted after the transaction commits.
            storage_deletes.append(old.source_path)
        await uow.documents.delete(replace_id)
