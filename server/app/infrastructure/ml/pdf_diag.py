"""PDF quality diagnostics before ingestion.

Checks performed:
  1. PDF type: text-based / scanned / mixed
  2. Extracted text volume per page
  3. Text quality (garbled characters, encoding issues)
  4. Final chunks that will be uploaded to Qdrant
  5. Problematic pages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz
from domain.value_objects.page_content_type import PageContentType

logger = logging.getLogger("default")


@dataclass(frozen=True)
class PdfQualityReport:
    """Результат оценки качества извлечения текста из PDF, уже ПОСЛЕ OCR.

    В отличие от `check_pdf` (диагностика "на глаз" до индексации), эта оценка
    смотрит на то, что реально получилось в `docs` после parse_pdf — то есть
    учитывает, помог ли OCR восстановить сканы, а не просто факт их наличия.
    """

    total_pages: int
    n_ok: int
    n_missing: int  # OCR не дал текста вообще
    n_garbled: int  # текст есть, но выглядит как мусор
    bad_ratio: float

    @property
    def is_low_quality(self) -> bool:
        return self.total_pages > 0 and self.bad_ratio > 0.3


def assess_pdf_extraction_quality(pdf_path: Path, extracted_docs: list) -> PdfQualityReport:
    """Сравнивает физическое число страниц PDF с тем, что реально извлеклось.

    Страница считается "плохой", если для неё нет ни одного Document в extracted_docs
    (текстовый слой пуст и OCR не справился), либо извлечённый текст мусорный
    (is_garbled). Это именно тот случай ">30% сканов/мусора, и OCR не помог",
    который нужно поймать перед индексацией.
    """
    doc = fitz.open(str(pdf_path))
    try:
        total_pages = len(doc)
    finally:
        doc.close()

    by_page: dict[int, str] = {}
    for d in extracted_docs:
        page_num = d.metadata.get("page")
        if page_num is not None:
            by_page[page_num] = by_page.get(page_num, "") + d.page_content

    n_missing = max(total_pages - len(by_page), 0)
    n_garbled = sum(1 for text in by_page.values() if is_garbled(text))
    # Pages with table content (pipe-separated) should not count as garbled
    n_table = sum(1 for text in by_page.values() if "|" in text and "---" in text)
    n_garbled = max(0, n_garbled - n_table)
    bad = n_missing + n_garbled
    bad_ratio = bad / total_pages if total_pages else 0.0

    return PdfQualityReport(
        total_pages=total_pages,
        n_ok=len(by_page) - n_garbled,
        n_missing=n_missing,
        n_garbled=n_garbled,
        bad_ratio=bad_ratio,
    )


def is_garbled(text: str) -> bool:
    """Эвристика: если >15% символов — нечитаемый мусор, это скан без OCR."""
    if not text:
        return False
    # Table content (pipes, dashes, numbers) is not garbled
    if "|" in text and "---" in text:
        return False
    total = len(text)
    normal = sum(1 for c in text if c.isalnum() or c in " .,;:!?-—\n\t()[]«»\"'")
    return (normal / total) < 0.6


def _page_has_tables(page) -> bool:
    """Check if a PDF page contains table structures."""
    try:
        tables = page.find_tables()
        return bool(tables and tables.tables)
    except Exception:
        return False


def classify_page(text: str, chars: int, page=None) -> tuple[str, str]:
    """Classify a page and return (type, description).

    Types: text, scan, garbled, empty, table.
    """
    if chars == 0:
        # Check if page has tables (tables may have no extracted text layer)
        if page is not None and _page_has_tables(page):
            return PageContentType.TABLE.value, "таблица (без текстового слоя)"
        return PageContentType.EMPTY.value, "пустая"
    if chars < 50:
        if page is not None and _page_has_tables(page):
            return PageContentType.TABLE.value, "таблица"
        return PageContentType.SCAN.value, f"скан/изображение ({chars} симв)"
    if is_garbled(text):
        # Check if it's actually a table (pipes + dashes)
        if page is not None and _page_has_tables(page):
            return PageContentType.TABLE.value, "таблица"
        return PageContentType.GARBLED.value, f"мусорный текст ({chars} симв)"
    return PageContentType.TEXT.value, f"текст ({chars} симв)"


def _log_page_stats(page_stats: list[dict], total_pages: int) -> None:
    """Iterate over pages and log type descriptions, stopping early if all text."""
    for i, p in enumerate(page_stats):
        ptype = p["type"]
        if ptype != PageContentType.TEXT.value or i < 3:
            logger.info("  стр.%d: %s", i + 1, p["desc"])
        elif i == 3 and all(p["type"] == PageContentType.TEXT.value for p in page_stats):
            remaining_text = sum(1 for p in page_stats[3:] if p["type"] == PageContentType.TEXT.value)
            logger.info(
                "  стр.4-%d: текст (%d страниц OK)", total_pages, remaining_text + len(page_stats) - 3
            )
            break


def _compute_type_counts(page_stats: list[dict]) -> dict:
    """Compute counts for each page type and total characters."""
    types = [p["type"] for p in page_stats]
    n_text = types.count(PageContentType.TEXT.value)
    n_scan = types.count(PageContentType.SCAN.value)
    n_garbled = types.count(PageContentType.GARBLED.value)
    n_empty = types.count(PageContentType.EMPTY.value)
    n_table = types.count(PageContentType.TABLE.value)
    total_chars = sum(p["chars"] for p in page_stats)
    return {
        "n_text": n_text,
        "n_scan": n_scan,
        "n_garbled": n_garbled,
        "n_empty": n_empty,
        "n_table": n_table,
        "total_chars": total_chars,
    }


def _log_diagnosis(
    n_text: int,
    n_scan: int,
    n_garbled: int,
    total_chars: int,
    pdf_path: Path,
) -> None:
    """Log diagnosis verdict based on page type counts."""
    logger.info("")
    logger.info("Диагноз:")

    if n_scan > n_text:
        logger.error("  PDF содержит преимущественно сканы — текст НЕ извлечётся")
        logger.info("    Решение: OCR через Tesseract (см. ниже)")
        ocr_hint(pdf_path)
    elif n_scan > 0:
        logger.warning("  PDF смешанный: %d текстовых + %d сканов", n_text, n_scan)
        logger.info("    Текстовые страницы индексируются нормально.")
        logger.info("    Для сканов нужен OCR.")
        ocr_hint(pdf_path)
    elif n_garbled > 0:
        logger.warning("  Мусорный текст на %d стр. — проблема со шрифтами PDF", n_garbled)
        logger.info("    Решение: конвертировать через LibreOffice или Ghostscript")
        convert_hint(pdf_path)
    elif total_chars < 500:
        logger.error("  Слишком мало текста — документ скорее всего пустой или изображение")
    else:
        logger.info("  PDF читается нормально, проблем не обнаружено")


def _log_chunk_stats(full_text: str, chunk_size: int, chunk_overlap: int) -> None:
    """Log chunk statistics from full extracted text."""
    chunks = simple_chunk(full_text, chunk_size, chunk_overlap)
    logger.info("")
    logger.info("Чанки (chunk_size=%d, overlap=%d):", chunk_size, chunk_overlap)
    logger.info("  Итого чанков: %d", len(chunks))
    if chunks:
        avg_chunk = sum(len(c) for c in chunks) / len(chunks)
        logger.info("  Средний размер: %.0f символов", avg_chunk)
        for i, ch in enumerate(chunks[:2], 1):
            preview = ch[:120].replace("\n", "↵")
            logger.info("  [%d] %s...", i, preview)


def check_pdf(pdf_path: Path, dump: bool = False, chunk_size: int = 512, chunk_overlap: int = 128) -> dict:
    logger.info("=" * 60)
    logger.info("%s  (%.0f KB)", pdf_path.name, pdf_path.stat().st_size / 1024)
    logger.info("-" * 60)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    logger.info("Страниц: %d", total_pages)

    page_stats = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        chars = len(text.strip())
        ptype, desc = classify_page(text, chars, page=page)
        page_stats.append(
            {
                "num": i + 1,
                "type": ptype,
                "chars": chars,
                "text": text,
                "desc": desc,
            }
        )

    _log_page_stats(page_stats, total_pages)
    doc.close()

    counts = _compute_type_counts(page_stats)
    n_text = counts["n_text"]
    n_scan = counts["n_scan"]
    n_garbled = counts["n_garbled"]
    total_chars = counts["total_chars"]

    logger.info("")
    logger.info("Итог:")
    logger.info("  Текстовых:    %d/%d", n_text, total_pages)
    if n_scan:
        logger.warning("  Сканов:       %d  ← нужен OCR", n_scan)
    if n_garbled:
        logger.warning("  Мусорных:     %d  ← проблема кодировки/шрифта", n_garbled)
    if counts["n_table"]:
        logger.info("  Таблиц:       %d", counts["n_table"])
    if counts["n_empty"]:
        logger.info("  Пустых:       %d", counts["n_empty"])
    avg_chars = total_chars // max(n_text, 1)
    logger.info("  Всего символов: %s", f"{total_chars:,}")
    logger.info("  Символов/стр (текст): ~%s", f"{avg_chars:,}")

    _log_diagnosis(n_text, n_scan, n_garbled, total_chars, pdf_path)

    full_text = "\n\n".join(
        p["text"] for p in page_stats if p["type"] == PageContentType.TEXT.value and p["text"].strip()
    )
    if full_text:
        _log_chunk_stats(full_text, chunk_size, chunk_overlap)

    if dump and full_text:
        logger.info("-" * 60)
        logger.info("ПОЛНЫЙ ТЕКСТ (первые 2000 символов):")
        logger.info("%s", full_text[:2000])
        if len(full_text) > 2000:
            logger.info("... (%d символов обрезано)", len(full_text) - 2000)

    return {
        "file": str(pdf_path),
        "pages": total_pages,
        "n_text": n_text,
        "n_scan": n_scan,
        "n_garbled": n_garbled,
        "total_chars": total_chars,
    }


def simple_chunk(text: str, size: int, overlap: int) -> list[str]:
    """Упрощённый сплиттер для диагностики."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def ocr_hint(pdf_path: Path):
    logger.info("  Хорошая новость: ingestion.py теперь сам делает OCR сканов через PaddleOCR.")
    logger.info("    Убедись, что OCR_ENABLED=true")
    logger.info("    Просто запусти индексацию: python main.py ingest file '%s'", pdf_path)
    logger.info("  Если результат PaddleOCR неудовлетворителен — попробуй Surya:")
    logger.info("    pip install surya-ocr")
    logger.info("    OCR_ENGINE=auto python main.py ingest file '%s'", pdf_path)


def convert_hint(pdf_path: Path):
    logger.info("  Конвертировать через Ghostscript:")
    logger.info("    sudo apt install ghostscript")
    logger.info(
        '    gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile="%s_fixed.pdf" "%s"', pdf_path.stem, pdf_path
    )
    logger.info("  Или через LibreOffice:")
    logger.info('    libreoffice --headless --convert-to pdf "%s" --outdir .', pdf_path)
