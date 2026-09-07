"""RTF preview strategy — decree-structured units or flat document fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from domain.value_objects.page_content_type import PageContentType, PreviewUnitKind

from application.services.pdf_diagnostic_service import DryRunPageResult
from infrastructure.ml.pdf_diag import is_garbled
from infrastructure.ml.ingestion import parse_rtf
from infrastructure.ml.rtf_decree_parser import parse_decree_rtf

if TYPE_CHECKING:
    from application.ports.domain_settings import DomainSettingsPort
    from infrastructure.domain_profile.registry import DomainProfileRegistry

logger = logging.getLogger("default")

_DECREE_FINGERPRINT_PREFIX_CHARS = 3000


class RtfPreviewStrategy:
    def __init__(
        self,
        domain_registry: "DomainProfileRegistry | None" = None,
        domain_settings: "DomainSettingsPort | None" = None,
    ) -> None:
        self._domain_registry = domain_registry
        self._domain_settings = domain_settings

    def supports(self, extension: str) -> bool:
        return extension == ".rtf"

    def _decree_fingerprint_hits(self, text: str) -> bool:
        if self._domain_registry is None or self._domain_settings is None:
            return False
        try:
            profile = self._domain_registry.get("decree")
        except KeyError:
            return False
        return profile.structural_fingerprint(text[:_DECREE_FINGERPRINT_PREFIX_CHARS])

    def analyze(self, path: Path) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        raw_text, _meta = parse_rtf(path)
        stripped = raw_text.strip()
        chars = len(stripped)

        if self._decree_fingerprint_hits(stripped):
            return self._analyze_decree(path, stripped)

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

    def _analyze_decree(
        self, path: Path, raw_text: str
    ) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        """Decree-structured preview: one unit per structural SplitUnit."""
        profile = self._domain_registry.get("decree")
        try:
            split_units, _doc_metadata = parse_decree_rtf(path, profile, self._domain_settings)
        except Exception:
            logger.exception("Decree preview failed for %s — falling back to flat preview", path.name)
            return self._analyze_flat(raw_text)

        types_count: dict[str, int] = {
            PageContentType.TEXT: 0,
            PageContentType.SCAN: 0,
            PageContentType.GARBLED: 0,
            PageContentType.EMPTY: 0,
            PageContentType.TABLE: 0,
            PageContentType.IMAGE_ONLY: 0,
        }
        units: list[DryRunPageResult] = []
        total_chars = 0
        for i, unit in enumerate(split_units, start=1):
            content = unit.content.strip()
            if not content:
                continue
            chars = len(content)
            total_chars += chars
            if chars == 0:
                ptype = PageContentType.EMPTY
            elif is_garbled(content):
                ptype = PageContentType.GARBLED
            else:
                ptype = PageContentType.TEXT
            types_count[ptype] += 1
            units.append(
                DryRunPageResult(
                    page=i,
                    type=ptype,
                    content_type=PageContentType.TEXT,
                    chars=chars,
                    preview=content[:200],
                    full_text=content,
                    unit_kind=PreviewUnitKind.SECTION,
                    label=unit.heading or unit.unit_kind,
                )
            )
        if not units:
            return self._analyze_flat(raw_text)
        return units, types_count, total_chars

    def _analyze_flat(self, raw_text: str) -> tuple[list[DryRunPageResult], dict[str, int], int]:
        stripped = raw_text.strip()
        chars = len(stripped)
        types_count = {
            PageContentType.TEXT: 1,
            PageContentType.SCAN: 0,
            PageContentType.GARBLED: 0,
            PageContentType.EMPTY: 0,
            PageContentType.TABLE: 0,
            PageContentType.IMAGE_ONLY: 0,
        }
        units = [
            DryRunPageResult(
                page=1,
                type=PageContentType.TEXT,
                content_type=PageContentType.TEXT,
                chars=chars,
                preview=stripped[:200],
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
