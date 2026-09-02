"""RTF preview strategy — flat text, single unit for the entire document."""

from __future__ import annotations

import logging
from pathlib import Path

from domain.value_objects.page_content_type import PageContentType, PreviewUnitKind

from application.services.pdf_diagnostic_service import DryRunPageResult
from infrastructure.ml.pdf_diag import is_garbled
from infrastructure.ml.ingestion import _parse_rtf

logger = logging.getLogger("default")


class RtfPreviewStrategy:
    def supports(self, extension: str) -> bool:
        return extension == ".rtf"

    def analyze(self, path: Path) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        text, _meta = _parse_rtf(path)
        stripped = text.strip()
        chars = len(stripped)

        if chars == 0:
            ptype = PageContentType.EMPTY
        elif is_garbled(stripped):
            ptype = PageContentType.GARBLED
        else:
            ptype = PageContentType.TEXT

        types_count: dict[str, int] = {
            PageContentType.TEXT: 0,
            PageContentType.SCAN: 0,
            PageContentType.GARBLED: 0,
            PageContentType.EMPTY: 0,
            PageContentType.TABLE: 0,
            PageContentType.IMAGE_ONLY: 0,
        }
        types_count[ptype] = 1

        preview = stripped[:200] if stripped else ""
        units = [
            DryRunPageResult(
                page=1,
                type=ptype,
                content_type=PageContentType.TEXT,
                chars=chars,
                preview=preview,
                full_text=stripped,
                unit_kind=PreviewUnitKind.DOCUMENT,
                label="Документ целиком",
            )
        ]

        return units, types_count, chars

    def ocr_problem_units(
        self,
        path: Path,
        units: list[DryRunPageResult],
        unit_ids: list[int],
    ) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        raise ValueError("OCR not supported for RTF — no access to embedded images")
