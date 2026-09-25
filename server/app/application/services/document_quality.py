"""Document quality assessment -- extraction quality checks for uploaded files.

Extracted from DocumentProcessor: the quality step is a pure function over
parser output and assessor ports, with no transaction or storage concerns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from application.ports.document_processing import (
        MetricsCollectorPort,
        PDFQualityAssessorPort,
        TextQualityAssessorPort,
    )
    from domain.value_objects.pdf_quality_report import PDFQualityReport

log = logging.getLogger("default")


@dataclass(frozen=True)
class QualityOutcome:
    """Result of the quality step: optional report + optional user-facing warning."""

    report: PDFQualityReport | None
    warning: str | None


def assess_document_quality(
    temp_path: Path,
    original_filename: str,
    document_id: int,
    docs: list,
    *,
    pdf_assessor: PDFQualityAssessorPort,
    text_quality_assessor: TextQualityAssessorPort,
    metrics: MetricsCollectorPort,
) -> QualityOutcome:
    """Assess document quality and return report + warning if issues found.

    Supports PDF, DOCX, and RTF formats.  Only the PDF branch produces a
    report object (other formats never persist a quality_score).
    """
    suffix = Path(original_filename).suffix.lower()
    warning_message = None

    if suffix == ".pdf":
        quality = pdf_assessor.assess(temp_path, docs)
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
        metrics.observe_pdf_pages("ok", quality.n_ok)
        metrics.observe_pdf_pages("missing", quality.n_missing)
        metrics.observe_pdf_pages("garbled", quality.n_garbled)
        metrics.observe_pdf_bad_ratio(quality.bad_ratio)
        return QualityOutcome(quality, warning_message)

    # Generic quality check for other formats (DOCX, RTF, TXT, MD)
    total_chars = sum(len(d.page_content) for d in docs)
    garbled_pages = sum(1 for d in docs if text_quality_assessor.is_garbled(d.page_content))
    empty_pages = sum(1 for d in docs if not d.page_content.strip())
    total_pages = len(docs)

    if total_pages == 0:
        return QualityOutcome(None, None)

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

    return QualityOutcome(None, warning_message)
