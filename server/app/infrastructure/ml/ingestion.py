"""Pure document parsers and text splitting -- no Qdrant, no storage, no registry.

Provides file-type-specific parsers (PDF via PyMuPDF, DOCX, Markdown, plain
text) and a LangChain-compatible text splitter.  All functions are pure:
they accept a file path and return ``langchain.schema.Document`` lists with
no side effects.
"""

import functools
import html
import logging
import re
from pathlib import Path

import docx
import fitz
import numpy as np
from config import settings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from paddleocr import PaddleOCR
from PIL import Image
from striprtf.striprtf import rtf_to_text

log = logging.getLogger("detailed")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("detailed")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path = Path(settings.data_dir) / "ingestion.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass

    return logger


# ---------------------------------------------------------------------------
# OCR — lazy-loaded via lru_cache (no global keyword)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_paddle_ocr():
    log.info("Loading PaddleOCR (lang=%s) ...", settings.ocr_lang_paddle)
    return PaddleOCR(
        use_angle_cls=True,
        lang=settings.ocr_lang_paddle,
        show_log=False,
    )


@functools.lru_cache(maxsize=1)
def _get_surya_predictors():
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor

    log.info("Loading Surya OCR ...")
    return (RecognitionPredictor(), DetectionPredictor())


def _ocr_image_paddle(image) -> str:
    ocr = _get_paddle_ocr()
    result = ocr.ocr(np.array(image), cls=True)
    lines = []
    for block in result or []:
        for entry in block or []:
            text = entry[1][0]
            if text and text.strip():
                lines.append(text.strip())
    return "\n".join(lines)


def _ocr_image_surya(image) -> str:
    rec_predictor, det_predictor = _get_surya_predictors()
    predictions = rec_predictor([image], [settings.ocr_lang_surya], det_predictor)
    lines = [line.text.strip() for line in predictions[0].text_lines if line.text.strip()]
    return "\n".join(lines)


def ocr_pdf_pages(doc, page_nums: list[int], filename: str) -> dict:
    results = {}
    zoom = settings.ocr_dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page_num in page_nums:
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        text = ""
        if settings.ocr_engine in ("paddleocr", "auto"):
            text = _ocr_image_paddle(img)

        if not text and settings.ocr_engine in ("surya", "auto"):
            text = _ocr_image_surya(img)

        results[page_num] = text

    return results


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------


def parse_pdf(file_path: Path) -> list[Document]:
    doc = fitz.open(str(file_path))
    pages = []

    ocr_pages_needed = []
    for page_num in range(1, len(doc) + 1):
        page = doc.load_page(page_num - 1)
        text = page.get_text("text").strip()
        if not text and settings.ocr_enabled:
            ocr_pages_needed.append(page_num)
        elif text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": page_num, "source": str(file_path)},
                )
            )

    if ocr_pages_needed:
        ocr_results = ocr_pdf_pages(doc, ocr_pages_needed, file_path.name)
        for page_num, text in ocr_results.items():
            if text:
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_num, "source": str(file_path)},
                    )
                )

    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------


def merge_pdf_pages(pages: list[Document]) -> list[Document]:
    """Merge per-page Documents from the same source into a single Document.

    Preserves page range in metadata (page_start, page_end, pages list).
    Prevents the splitter from breaking text at page boundaries.
    """
    if len(pages) <= 1:
        return pages

    # Group by source
    by_source: dict[str, list[Document]] = {}
    for p in pages:
        src = p.metadata.get("source", "")
        by_source.setdefault(src, []).append(p)

    merged = []
    for src, group in by_source.items():
        page_nums = sorted(p.metadata.get("page", i + 1) for i, p in enumerate(group))
        merged_text = "\n\n".join(p.page_content for p in group)
        merged.append(
            Document(
                page_content=merged_text,
                metadata={
                    **group[0].metadata,
                    "page_start": page_nums[0],
                    "page_end": page_nums[-1],
                    "pages": page_nums,
                },
            )
        )
    return merged


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into chunks using structure-aware separators.

    Separators are ordered by priority: articles/sections first, then paragraphs,
    lines, words, characters. This prevents the splitter from breaking inside
    numbered articles, legal clauses, or structured content.
    """
    separators = [
        "\nСтатья ",
        "\nРаздел ",
        "\nПункт ",
        "\n\n",
        "\n",
        " ",
        "",
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=separators,
    )
    chunks = splitter.split_documents(docs)
    log.info("Split %d documents into %d chunks", len(docs), len(chunks))
    return chunks


LEGAL_SEPARATORS = [
    "\nГлава ",
    "\nРаздел ",
    "\nЧасть ",
    "\nСтатья ",
    "\n§ ",
    "\nПункт ",
    "\n\n",
    "\n",
    " ",
    "",
]


def split_documents_legal(docs: list[Document]) -> list[Document]:
    """Split legal documents into chunks with larger size and legal-aware separators.

    Uses legal_chunk_size (default 1000) and legal_chunk_overlap (default 250)
    to keep articles/clauses intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.legal_chunk_size,
        chunk_overlap=settings.legal_chunk_overlap,
        length_function=len,
        separators=LEGAL_SEPARATORS,
    )
    chunks = splitter.split_documents(docs)
    log.info("Split %d legal documents into %d chunks", len(docs), len(chunks))
    return chunks


_ARTICLE_RE = re.compile(r"Статья\s+(\d+[\.\d]*)")


def extract_article_number(chunk_text: str) -> str | None:
    """Extract article number from chunk text if present."""
    m = _ARTICLE_RE.search(chunk_text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Parsers for non-PDF formats
# ---------------------------------------------------------------------------


def _has_page_break(paragraph) -> bool:
    """Check if a paragraph contains a manual page break."""
    from docx.oxml.ns import qn

    for run in paragraph.runs:
        for br in run._element.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    return False


def _parse_docx(file_path: Path) -> tuple[str, dict]:
    """Parse DOCX and return (text, metadata).

    Detects page breaks via <w:br w:type="page"/> to provide page metadata.
    """
    doc = docx.Document(str(file_path))
    parts = []
    page_numbers: list[int] = []
    current_page = 1

    for p in doc.paragraphs:
        if _has_page_break(p):
            current_page += 1
        if not p.text.strip():
            continue
        page_numbers.append(current_page)
        parts.append(p.text)

    for table in doc.tables:
        parts.append("")  # blank line before table
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))

    metadata: dict = {}
    if page_numbers:
        metadata["page_start"] = page_numbers[0]
        metadata["page_end"] = page_numbers[-1]
        metadata["pages"] = sorted(set(page_numbers))
    return "\n".join(parts), metadata


def _parse_rtf(file_path: Path) -> tuple[str, dict]:
    """Parse RTF and return (text, metadata)."""
    return rtf_to_text(file_path.read_text(encoding="utf-8", errors="replace")), {}


_MD_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_DATE_IN_FILENAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _clean_markdown_text(text: str) -> str:
    """Общая очистка markdown-разметки — используется и flat-парсером, и секционным."""
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"[image: \1](\2)", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`~]+", "", text)
    text = html.unescape(text)
    return text.strip()


def parse_markdown_sections(file_path: Path) -> list[tuple[str | None, str]]:
    """Разбивает markdown на (заголовок, контент) по структуре заголовков.

    Контент между двумя заголовками относится к предыдущему заголовку. Текст до
    первого заголовка (если есть) получает heading=None.
    """
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    matches = list(_MD_HEADER_RE.finditer(raw))

    if not matches:
        return [(None, _clean_markdown_text(raw))]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        lead = _clean_markdown_text(raw[: matches[0].start()])
        if lead:
            sections.append((None, lead))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = _clean_markdown_text(raw[start:end])
        if content:
            sections.append((heading, content))

    return sections


def parse_docx_sections(file_path: Path) -> list[tuple[str | None, str]]:
    """Разбивает DOCX на (заголовок, контент) по параграфам со стилем Heading*/Title.

    Also detects page breaks to track page numbers within each section.
    """
    doc = docx.Document(str(file_path))
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_heading, content))

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style_name = (p.style.name or "").lower() if p.style else ""
        if style_name.startswith("heading") or style_name == "title":
            _flush()
            current_heading = text
            current_lines = []
        else:
            current_lines.append(text)
    _flush()

    table_lines = []
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            table_lines.append(" | ".join(cells))
    if table_lines:
        sections.append((None, "\n".join(table_lines)))

    if not sections:
        text, _meta = _parse_docx(file_path)
        return [(None, text)]
    return sections


def extract_date_from_filename(filename: str) -> str | None:
    """Ищет дату вида YYYY-MM-DD в имени файла — простая эвристика для метаданных."""
    m = _DATE_IN_FILENAME_RE.search(filename)
    return m.group(1) if m else None


def _parse_markdown(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return _clean_markdown_text(text)


def _parse_txt(file_path: Path) -> tuple[str, dict]:
    """Parse TXT and return (text, metadata)."""
    return file_path.read_text(encoding="utf-8", errors="replace"), {}


PARSERS = {
    ".docx": _parse_docx,
    ".doc": _parse_docx,
    ".rtf": _parse_rtf,
    ".md": _parse_markdown,
    ".txt": _parse_txt,
}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_pdf_text(text: str) -> str:
    """Clean extracted PDF text: fix hyphenation, collapse whitespace, remove decorative lines."""
    # Fix hyphenation at line breaks
    text = re.sub(r"-\n", "", text)
    # Collapse multiple whitespace to single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove decorative separator lines (---, ===, *** etc.)
    text = re.sub(r"^[•\-=~*]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()
