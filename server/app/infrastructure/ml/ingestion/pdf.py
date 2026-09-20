"""PDF parsing — PyMuPDF-based text/table extraction with OCR fallback."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from langchain.schema import Document

from config import settings
from domain.value_objects.page_content_type import PageContentType
from infrastructure.ml.ingestion.ocr import ocr_pdf_pages
from infrastructure.ml.ingestion.utils import clean_pdf_text

log = logging.getLogger("detailed")

# A gap between adjacent block left-edges larger than this fraction of the
# page width is treated as a column break (multi-column layout detection).
_COLUMN_GAP_RATIO = 0.08

# Header/footer margin zones, as a fraction of page height. Only blocks
# whose bounding box falls entirely within these zones are ever considered
# for boilerplate removal — this is what keeps ordinary body text (which
# may legitimately repeat similar phrasing with different numbers, e.g.
# "Section 1" .. "Section 5") from being misclassified as a running header.
_HEADER_ZONE_RATIO = 0.06
_FOOTER_ZONE_RATIO = 0.09

# Running headers/footers must repeat on at least this fraction of pages
# (and a minimum absolute count) before being treated as boilerplate.
_BOILERPLATE_MIN_RATIO = 0.6
_BOILERPLATE_MIN_PAGES = 4
_DIGITS_RE = re.compile(r"\d+")


@dataclass
class PageResult:
    """Return type for _process_page: collects per-page outputs."""

    text_doc: Document | None = None
    ocr_needed: bool = False
    text_for_comparison: str = ""
    meta: dict = field(default_factory=dict)
    table_docs: list[Document] = field(default_factory=list)


def pymupdf_table_to_markdown(table) -> str:
    """Convert a PyMuPDF table to markdown format."""
    try:
        data = table.extract()
    except Exception:
        return ""
    if not data or len(data) < 1:
        return ""

    # Normalize column count
    max_cols = max(len(row) for row in data)
    rows = []
    for row in data:
        normalized = [str(cell).strip().replace("|", "\\|") if cell else "" for cell in row]
        while len(normalized) < max_cols:
            normalized.append("")
        rows.append(normalized)

    header = "| " + " | ".join(rows[0]) + " |"
    separator = "|" + "|".join(["---"] * max_cols) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator] + body) if body else header


def _order_blocks_columnwise(blocks: list, page_width: float) -> list[str]:
    """Order a set of text blocks for correct reading order.

    Single-column layouts are read top-to-bottom (same as a plain row sort).
    For multi-column layouts, blocks are clustered into columns by their
    left-edge x-position (a gap larger than ~8% of the page width signals a
    column break), each column is read top-to-bottom, and columns are
    concatenated left-to-right — a naive row-based sort would otherwise
    interleave lines from adjacent columns into nonsense.
    """
    if not blocks:
        return []
    if len(blocks) == 1:
        return [blocks[0][4].strip()]

    gap_threshold = (page_width or 1.0) * _COLUMN_GAP_RATIO
    xs = sorted(b[0] for b in blocks)
    boundaries = [xs[0]]
    for prev, cur in zip(xs, xs[1:], strict=False):
        if cur - prev > gap_threshold:
            boundaries.append(cur)

    def _column_of(x0: float) -> int:
        col = 0
        for i, boundary in enumerate(boundaries):
            if x0 >= boundary:
                col = i
        return col

    columns: dict[int, list] = {}
    for b in blocks:
        columns.setdefault(_column_of(b[0]), []).append(b)

    ordered: list[str] = []
    for col in sorted(columns):
        col_blocks = sorted(columns[col], key=lambda b: b[1])
        ordered.extend(b[4].strip() for b in col_blocks if b[4].strip())
    return ordered


def _classify_page_blocks(page) -> tuple[list, list, list]:
    """Split a page's text blocks into (header_blocks, body_blocks, footer_blocks).

    A block counts as header/footer only if it lies entirely within the
    top/bottom margin zone — this is a position-based classification, not a
    content-based one, so it never touches ordinary body paragraphs.
    """
    height = page.rect.height or 1.0
    header_end = height * _HEADER_ZONE_RATIO
    footer_start = height * (1 - _FOOTER_ZONE_RATIO)

    header, body, footer = [], [], []
    for b in page.get_text("blocks"):
        if len(b) <= 4 or not b[4].strip():
            continue
        y0, y1 = b[1], b[3]
        if y1 <= header_end:
            header.append(b)
        elif y0 >= footer_start:
            footer.append(b)
        else:
            body.append(b)
    return header, body, footer


def _extract_page_tables(page, file_path: Path, page_num: int) -> list[Document]:
    """Detect and extract tables from a page. Returns list of table Documents."""
    docs: list[Document] = []
    try:
        tables = page.find_tables()
        if tables and tables.tables:
            for table in tables.tables:
                md_table = pymupdf_table_to_markdown(table)
                if md_table:
                    docs.append(
                        Document(
                            page_content=md_table,
                            metadata={
                                "page": page_num,
                                "source": str(file_path),
                                "content_type": PageContentType.TABLE.value,
                            },
                        )
                    )
    except Exception:
        log.debug("Failed to extract tables from page %s", page_num, exc_info=True)
    return docs


def _detect_tabular_heuristic(text: str, page_num: int) -> None:
    """Log warning if text has tabular-looking lines but find_tables() found nothing."""
    lines = text.split("\n")
    tabular_lines = sum(
        1 for line in lines if line.count("\t") >= 2 or (len(re.findall(r"  {3,}", line)) > 0)
    )
    if tabular_lines > 3:
        log.warning(
            "Page %d: %d tabular-looking lines but find_tables() found nothing",
            page_num,
            tabular_lines,
        )


def _should_ocr(text: str, min_chars: int, ocr_enabled: bool) -> bool:
    """Return True if OCR should be triggered based on text length and settings."""
    if not text and ocr_enabled:
        return True
    if text and len(text.strip()) < min_chars and ocr_enabled:
        return True
    return False


def _select_best_text(existing_text: str | None, ocr_text: str) -> str:
    """Choose between OCR text and existing short text layer, preferring longer."""
    if existing_text:
        if len(ocr_text) > len(existing_text) * 1.5:
            return ocr_text
        return clean_pdf_text(existing_text) or ocr_text
    return ocr_text


def _process_page_text(text: str, tables_found: bool, page_num: int, file_path: Path) -> Document | None:
    if text and not tables_found:
        _detect_tabular_heuristic(text, page_num)
    if not text or tables_found:
        return None
    cleaned = clean_pdf_text(text)
    if not cleaned:
        return None
    return Document(page_content=cleaned, metadata={"page": page_num, "source": str(file_path)})


def _process_ocr_result(
    page_num: int,
    ocr_text: str,
    text_to_compare: dict,
    file_path: Path,
) -> Document | None:
    if not ocr_text:
        return None
    ocr_text = clean_pdf_text(ocr_text)
    if not ocr_text:
        return None
    final_text = _select_best_text(text_to_compare.get(page_num), ocr_text)
    return Document(page_content=final_text, metadata={"page": page_num, "source": str(file_path)})


def _find_boilerplate_patterns(
    classified_pages: list[tuple[list, list, list]],
) -> set[str]:
    """Determine which header/footer line patterns repeat across most pages.

    Only header/footer-zone blocks (see _classify_page_blocks) are ever
    candidates. Digit runs are normalized ("Page 3" / "Page 4" -> "Page #")
    so a changing page number still counts as the same recurring pattern,
    without touching anything outside the margin zones.
    """
    if len(classified_pages) < _BOILERPLATE_MIN_PAGES:
        return set()

    counts: Counter[str] = Counter()
    for header, _body, footer in classified_pages:
        seen_this_page = set()
        for b in header + footer:
            for line in b[4].strip().split("\n"):
                norm = _DIGITS_RE.sub("#", line.strip())
                if norm:
                    seen_this_page.add(norm)
        counts.update(seen_this_page)

    threshold = max(_BOILERPLATE_MIN_PAGES, int(len(classified_pages) * _BOILERPLATE_MIN_RATIO))
    return {norm for norm, count in counts.items() if count >= threshold}


def _filter_boilerplate_blocks(blocks: list, boilerplate: set[str]) -> list:
    if not boilerplate:
        return blocks
    kept = []
    for b in blocks:
        lines = [ln.strip() for ln in b[4].strip().split("\n") if ln.strip()]
        if lines and all(_DIGITS_RE.sub("#", ln) in boilerplate for ln in lines):
            continue  # entire block is recurring boilerplate
        kept.append(b)
    return kept


def _extract_doc_metadata(doc) -> dict:
    """Pull PDF document-info metadata (title/author/subject/keywords) if present.

    PyMuPDF exposes this as doc.metadata — a plain dict with string values,
    often empty strings when the producer didn't set them. Only non-empty
    values are kept so we never overwrite a real value with "".
    """
    meta = doc.metadata or {}
    result: dict = {}
    if meta.get("title"):
        result["doc_title"] = meta["title"]
    if meta.get("author"):
        result["doc_author"] = meta["author"]
    if meta.get("subject"):
        result["doc_subject"] = meta["subject"]
    if meta.get("keywords"):
        result["doc_keywords"] = meta["keywords"]
    result["page_count"] = len(doc)
    return result


def _compute_text_quality(page) -> float:
    """Estimate text extraction quality as ratio of text chars to page area."""
    try:
        page_area = page.rect.width * page.rect.height
        if page_area <= 0:
            return 0.0
        text = page.get_text("text")
        char_count = len(text.strip())
        # Rough heuristic: ~0.02 chars per pixel is a "full" page of text
        expected_chars = page_area * 0.02
        return min(1.0, char_count / expected_chars) if expected_chars > 0 else 0.0
    except Exception:
        return 0.0


def _process_page(
    doc: fitz.Document,
    page_num: int,
    classified: list[tuple[list, list, list]],
    boilerplate: set[str],
    file_path: Path,
    ocr_enabled: bool,
    min_chars: int,
) -> PageResult:
    """Process a single page: extract text, tables, decide if OCR is needed."""
    page = doc.load_page(page_num - 1)
    header, body, footer = classified[page_num - 1]

    image_list = page.get_images(full=True)
    page_meta = {
        "has_images": bool(image_list),
        "image_count": len(image_list),
        "text_quality": round(_compute_text_quality(page), 3),
    }

    table_docs = _extract_page_tables(page, file_path, page_num)

    kept_header = _filter_boilerplate_blocks(header, boilerplate)
    kept_footer = _filter_boilerplate_blocks(footer, boilerplate)
    header_lines = [b[4].strip() for b in sorted(kept_header, key=lambda b: b[1]) if b[4].strip()]
    footer_lines = [b[4].strip() for b in sorted(kept_footer, key=lambda b: b[1]) if b[4].strip()]
    body_lines = _order_blocks_columnwise(body, page.rect.width)
    text = "\n".join(header_lines + body_lines + footer_lines)

    if _should_ocr(text, min_chars, ocr_enabled):
        return PageResult(
            ocr_needed=True,
            text_for_comparison=text,
            meta=page_meta,
            table_docs=table_docs,
        )

    text_doc = _process_page_text(text, bool(table_docs), page_num, file_path)
    if text_doc:
        text_doc.metadata.update(page_meta)
    return PageResult(text_doc=text_doc, meta=page_meta, table_docs=table_docs)


def _process_ocr_batch(
    doc: fitz.Document,
    ocr_pages_needed: list[int],
    text_to_compare: dict[int, str],
    file_path: Path,
) -> list[Document]:
    """Run OCR on flagged pages and return the resulting Documents."""
    pages: list[Document] = []
    ocr_results = ocr_pdf_pages(doc, ocr_pages_needed, file_path.name)
    for page_num, ocr_text in ocr_results.items():
        ocr_doc = _process_ocr_result(page_num, ocr_text, text_to_compare, file_path)
        if ocr_doc:
            ocr_doc.metadata["has_images"] = True
            ocr_doc.metadata["image_count"] = len(doc.load_page(page_num - 1).get_images(full=True))
            ocr_doc.metadata["text_quality"] = 0.0
            pages.append(ocr_doc)
    return pages


def _apply_document_metadata(pages: list[Document], doc_metadata: dict, is_scanned: bool) -> None:
    """Stamp doc-level metadata and scanned flag onto every page."""
    if doc_metadata:
        for p in pages:
            p.metadata.update(doc_metadata)
    if is_scanned:
        for p in pages:
            p.metadata["is_scanned"] = True


def parse_pdf(file_path: Path) -> list[Document]:
    doc = fitz.open(str(file_path))
    n_pages = len(doc)
    doc_metadata = _extract_doc_metadata(doc)

    classified = [_classify_page_blocks(doc.load_page(i)) for i in range(n_pages)]
    boilerplate = _find_boilerplate_patterns(classified)
    if boilerplate:
        log.info("Detected %d recurring header/footer line pattern(s)", len(boilerplate))

    pages: list[Document] = []
    ocr_pages_needed: list[int] = []
    text_to_compare: dict[int, str] = {}
    ocr_page_set: set[int] = set()

    for page_num in range(1, n_pages + 1):
        result = _process_page(
            doc,
            page_num,
            classified,
            boilerplate,
            file_path,
            settings.ocr_enabled,
            settings.ocr_min_chars,
        )
        pages.extend(result.table_docs)
        if result.ocr_needed:
            ocr_pages_needed.append(page_num)
            ocr_page_set.add(page_num)
            if result.text_for_comparison:
                text_to_compare[page_num] = result.text_for_comparison
        elif result.text_doc:
            pages.append(result.text_doc)

    if ocr_pages_needed:
        pages.extend(_process_ocr_batch(doc, ocr_pages_needed, text_to_compare, file_path))

    is_scanned = len(ocr_page_set) >= n_pages * 0.8 if n_pages > 0 else False
    doc.close()
    _apply_document_metadata(pages, doc_metadata, is_scanned)
    return pages
