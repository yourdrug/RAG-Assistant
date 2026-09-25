"""Application service for processing uploaded documents end-to-end.

Orchestrates the pipeline: download from storage, parse, split, persist
chunk metadata to Postgres, and enqueue vector-store operations via the
Transactional Outbox pattern.  The outbox dispatcher applies changes to
Qdrant asynchronously after the Postgres transaction commits.

The steps themselves live in sibling modules (document_quality,
domain_classification, document_persistence) -- this class only wires them
together and owns the per-run state (ProcessingContext).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from application.dto.versioning_dto import VersioningResult
from application.ports.document_processing import (
    ContentExtractorPort,
    MetricsCollectorPort,
    PDFQualityAssessorPort,
    TextQualityAssessorPort,
)
from application.ports.file_storage import FileStorage
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.document_conflict_resolver import resolve_conflict
from application.services.document_persistence import persist_document_result
from application.services.document_quality import assess_document_quality
from application.services.domain_classification import classify_document_text
from domain.repositories.vector_store_repository import VectorStoreRepository
from domain.services.document_domain_classifier import classify_document_domain
from domain.services.document_parser import DocumentParser, DocumentSplitter
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.pdf_quality_report import PDFQualityReport

if TYPE_CHECKING:
    from domain.domain_profile.settings_port import DomainSettingsPort
    from application.services.act_versioning_service import ActVersioningService
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.entities.raw_document import RawDocument

log = logging.getLogger("default")

_EMPTY_VERSIONING = VersioningResult(
    domain_metadata=None, act_version_id=None, act_id=None, effective_from=None, warning=None
)


@dataclass
class ProcessingContext:
    """Mutable state of a single ``process()`` run.

    Inputs are fixed by ``process()``; pipeline steps append warnings and fill
    runtime fields instead of threading 10+ positional arguments through
    every helper.
    """

    # -- inputs (set once) --
    document_id: int
    storage_key: str
    original_filename: str
    visibility: str
    owner_id: int | None
    group_id: int | None
    replace_id: int | None
    doc_domain: str | None = None
    # -- runtime state --
    temp_path: Path | None = None
    docs: list = field(default_factory=list)
    quality: PDFQualityReport | None = None
    warnings: list[str] = field(default_factory=list)
    raw_chunks: list | None = None
    storage_deletes: list[str] = field(default_factory=list)
    status: str = DocumentStatus.FAILED.value
    versioning: VersioningResult = field(default_factory=lambda: _EMPTY_VERSIONING)

    @property
    def warning_message(self) -> str | None:
        """Warnings joined in step order (quality → ambiguous → versioning)."""
        return "\n".join(self.warnings) or None


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
        self._vector_store = vector_store_repo
        self._file_storage = file_storage
        self._parser = document_parser
        self._splitter = document_splitter
        self._extractor = content_extractor
        self._pdf_assessor = pdf_quality_assessor
        self._text_quality_assessor = text_quality_assessor
        self._metrics = metrics
        self._domain_marker_threshold = domain_marker_threshold
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings
        self._act_versioning_service = act_versioning_service

    @staticmethod
    def _enrich_chunk_with_section(rc: "RawDocument", doc_domain: str) -> None:
        section = rc.metadata.get("section")
        if section:
            rc.page_content = f"[Раздел: {section}]\n{rc.page_content}"

    def _classify_domain(self, full_text: str, document_id: int) -> tuple[str, str | None]:
        """Registry-based classification with legacy-marksman fallback.

        Returns (domain_key, ambiguous_warning | None). Ambiguous results are
        never swallowed — they surface in documents.warning_message.
        """
        classification = classify_document_text(
            full_text,
            domain_registry=self._domain_registry,
            domain_settings=self._domain_settings,
            fallback_threshold=self._domain_marker_threshold,
            legacy_classifier=classify_document_domain,
        )
        if classification.used_registry:
            self._metrics.observe_domain_classification(
                classification.domain_key, "document", classification.confidence
            )
        if classification.warning:
            self._metrics.inc_domain_ambiguous(classification.ambiguous_candidates)
            log.warning("Ambiguous classification for doc %d: %s", document_id, classification.scores_str)
        return classification.domain_key, classification.warning

    async def _handle_versioning(self, ctx: ProcessingContext, full_text: str) -> None:
        """Delegate to the unified versioning mechanism (shared with CLI ingestion)."""
        result = _EMPTY_VERSIONING
        if self._domain_registry is not None and self._act_versioning_service is not None:
            try:
                profile = self._domain_registry.get(ctx.doc_domain or "")
            except KeyError:
                profile = None
            if profile is not None:
                result = await self._act_versioning_service.process_document_versioning(
                    profile, ctx.document_id, full_text
                )
        ctx.versioning = result
        if result.warning:
            ctx.warnings.append(result.warning)

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

    def _finalize_processing(self, ctx: ProcessingContext, t_start: float) -> None:
        self._metrics.inc_documents(ctx.status)
        self._metrics.observe_duration(ctx.status, time.monotonic() - t_start)
        if ctx.temp_path is not None:
            ctx.temp_path.unlink(missing_ok=True)
        if ctx.raw_chunks:
            self._metrics.inc_chunks(len(ctx.raw_chunks))

    async def _classify_and_resolve_conflict(self, ctx: ProcessingContext, full_text: str) -> None:
        """Classify domain (if needed) and resolve async conflict for replacement.

        Fills ctx.doc_domain, appends the ambiguity warning, and collects
        replaced objects into ctx.storage_deletes (S3 cleanup happens after
        the persist transaction commits).
        """
        if ctx.doc_domain is None:
            doc_domain, ambiguous_warning = self._classify_domain(full_text, ctx.document_id)
            ctx.doc_domain = doc_domain
            if ambiguous_warning:
                ctx.warnings.append(ambiguous_warning)
            log.info("Auto-detected doc_domain=%s for doc %d", doc_domain, ctx.document_id)

        if ctx.replace_id is not None:
            new_doc = await self._get_document(ctx.document_id)
            old_doc = await self._get_document(ctx.replace_id)
            if new_doc and old_doc:
                profile = self._domain_registry.get(ctx.doc_domain) if self._domain_registry else None
                old_source_path = await resolve_conflict(
                    self._uow_factory,
                    new_doc,
                    old_doc,
                    profile,
                )
                if old_source_path:
                    ctx.storage_deletes.append(old_source_path)

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
            ctx.docs = self._parser.parse(ctx.temp_path)

            if not ctx.docs:
                raise RuntimeError(
                    "Текст не извлечён — документ похож на скан, и OCR не смог распознать содержимое."
                )

            outcome = assess_document_quality(
                ctx.temp_path,
                original_filename,
                document_id,
                ctx.docs,
                pdf_assessor=self._pdf_assessor,
                text_quality_assessor=self._text_quality_assessor,
                metrics=self._metrics,
            )
            ctx.quality = outcome.report
            if outcome.warning:
                ctx.warnings.append(outcome.warning)

            full_text = "\n".join(d.page_content for d in ctx.docs)

            await self._classify_and_resolve_conflict(ctx, full_text)
            doc_domain = ctx.doc_domain
            if doc_domain is None:
                raise RuntimeError(f"domain classification produced no domain for doc {document_id}")

            self._attach_metadata_to_docs(ctx.docs, original_filename, self._extractor)

            ctx.raw_chunks = self._splitter.split(ctx.docs, domain=doc_domain)
            for rc in ctx.raw_chunks:
                self._enrich_chunk_with_section(rc, doc_domain)

            await self._handle_versioning(ctx, full_text)

            current_doc = await self._get_document(document_id)
            if current_doc is None:
                log.info("Document %d was deleted during processing — aborting", document_id)
                return

            persisted = await persist_document_result(
                self._uow_factory,
                document_id=document_id,
                original_filename=original_filename,
                raw_chunks=ctx.raw_chunks,
                visibility=visibility,
                owner_id=owner_id,
                group_id=group_id,
                doc_domain=doc_domain,
                domain_metadata=ctx.versioning.domain_metadata,
                act_version_id=ctx.versioning.act_version_id,
                act_id=ctx.versioning.act_id,
                effective_from=ctx.versioning.effective_from,
                replace_id=replace_id,
                warning_message=ctx.warning_message,
                quality=ctx.quality,
                storage_deletes=ctx.storage_deletes,
                domain_registry=self._domain_registry,
            )
            if not persisted:
                return

            ctx.status = DocumentStatus.INDEXING.value

        except Exception as e:
            await self._handle_processing_failure(document_id, e)
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
