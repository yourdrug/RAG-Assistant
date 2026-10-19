"""Per-run state passed between uploaded-document processing steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from domain.value_objects.document_status import DocumentStatus


if TYPE_CHECKING:
    from pathlib import Path

    from domain.entities.raw_document import RawDocument


@dataclass
class ProcessingContext:
    """Mutable state of a single ``DocumentProcessor.process()`` run.

    The coordinator owns lifecycle state and warning order. Content preparation,
    classification and version planning return their results explicitly.
    """

    document_id: int
    replace_id: int | None
    doc_domain: str | None = None
    temp_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    raw_chunks: list[RawDocument] | None = None
    storage_deletes: list[str] = field(default_factory=list)
    status: DocumentStatus = DocumentStatus.FAILED

    @property
    def warning_message(self) -> str | None:
        """Warnings joined in step order (quality → ambiguous → versioning)."""
        return "\n".join(self.warnings) or None
