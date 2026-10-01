"""Per-run state passed between uploaded-document processing steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from application.dto.versioning_dto import VersioningPlan, VersioningResult
from domain.entities.raw_document import RawDocument
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.pdf_quality_report import PDFQualityReport

if TYPE_CHECKING:
    from domain.domain_profile.protocol import DomainProfile


def empty_versioning_result() -> VersioningResult:
    """Create an independent versioning result for a document without a version."""
    return VersioningResult(None, None, None, None, None)


@dataclass
class ProcessingContext:
    """Mutable state of a single ``DocumentProcessor.process()`` run.

    Inputs are set by the coordinator. Steps fill runtime fields and append
    warnings in processing order; contexts are never shared between runs.
    """

    document_id: int
    storage_key: str
    original_filename: str
    visibility: str
    owner_id: int | None
    group_id: int | None
    replace_id: int | None
    doc_domain: str | None = None
    temp_path: Path | None = None
    docs: list[RawDocument] = field(default_factory=list)
    quality: PDFQualityReport | None = None
    warnings: list[str] = field(default_factory=list)
    raw_chunks: list[RawDocument] | None = None
    storage_deletes: list[str] = field(default_factory=list)
    status: str = DocumentStatus.FAILED.value
    versioning: VersioningResult = field(default_factory=empty_versioning_result)
    versioning_plan: VersioningPlan | None = None
    versioning_profile: DomainProfile | None = None

    @property
    def warning_message(self) -> str | None:
        """Warnings joined in step order (quality → ambiguous → versioning)."""
        return "\n".join(self.warnings) or None
