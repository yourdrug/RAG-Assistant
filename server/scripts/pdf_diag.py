"""Local PDF diagnostics using the ingestion chunker."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import fitz
from langchain.schema import Document

from domain.knowledge.enums import PageContentType
from infrastructure.ml.guardrails.pdf_adapter import classify_page
from infrastructure.ml.ingestion.text_chunks import text_document_chunks


logger = logging.getLogger(__name__)
FIRST_TEXT_PAGES = 3
MIN_TOTAL_CHARS = 500
DUMP_LIMIT = 2000


def log_page_stats(page_stats: list[dict], total_pages: int) -> None:
    """Iterate over pages and log type descriptions, stopping early if all text."""
    for i, p in enumerate(page_stats):
        ptype = p["type"]
        if ptype != PageContentType.TEXT.value or i < FIRST_TEXT_PAGES:
            logger.info("  стр.%d: %s", i + 1, p["desc"])
        elif i == FIRST_TEXT_PAGES and all(p["type"] == PageContentType.TEXT.value for p in page_stats):
            remaining_text = sum(1 for p in page_stats[3:] if p["type"] == PageContentType.TEXT.value)
            logger.info(
                "  стр.4-%d: текст (%d страниц OK)", total_pages, remaining_text + len(page_stats) - 3
            )
            break


def compute_type_counts(page_stats: list[dict]) -> dict:
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


def log_diagnosis(
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
        logger.error("  PDF contains mostly scans; OCR is required")
        logger.info("    Решение: OCR через Tesseract (см. ниже)")
        ocr_hint(pdf_path)
    elif n_scan > 0:
        logger.warning("  PDF смешанный: %d текстовых + %d сканов", n_text, n_scan)
        logger.info("    Текстовые страницы индексируются нормально.")
        logger.info("    Для сканов нужен OCR.")
        ocr_hint(pdf_path)
    elif n_garbled > 0:
        logger.warning("  Garbled text on %d pages; check PDF fonts", n_garbled)
        logger.info("    Решение: конвертировать через LibreOffice или Ghostscript")
        convert_hint(pdf_path)
    elif total_chars < MIN_TOTAL_CHARS:
        logger.error("  Слишком мало текста — документ скорее всего пустой или изображение")
    else:
        logger.info("  PDF читается нормально, проблем не обнаружено")


def log_chunk_stats(full_text: str, chunk_size: int, chunk_overlap: int) -> None:
    """Log chunk statistics from full extracted text."""
    chunks = [
        doc.page_content
        for doc in text_document_chunks(Document(page_content=full_text), chunk_size, chunk_overlap)
    ]
    logger.info("")
    logger.info("Чанки (chunk_size=%d, overlap=%d):", chunk_size, chunk_overlap)
    logger.info("  Итого чанков: %d", len(chunks))
    if chunks:
        avg_chunk = sum(len(c) for c in chunks) / len(chunks)
        logger.info("  Средний размер: %.0f символов", avg_chunk)
        for i, ch in enumerate(chunks[:2], 1):
            preview = ch[:120].replace("\n", "↵")
            logger.info("  [%d] %s...", i, preview)


def log_summary(counts: dict, total_pages: int) -> None:
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
    logger.info("  Total characters: %s", f"{total_chars:,}")
    logger.info("  Символов/стр (текст): ~%s", f"{avg_chars:,}")


def check_pdf(pdf_path: Path, *, dump: bool = False, chunk_size: int = 512, chunk_overlap: int = 128) -> dict:
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

    log_page_stats(page_stats, total_pages)
    doc.close()

    counts = compute_type_counts(page_stats)
    n_text = counts["n_text"]
    n_scan = counts["n_scan"]
    n_garbled = counts["n_garbled"]
    total_chars = counts["total_chars"]

    log_summary(counts, total_pages)

    log_diagnosis(n_text, n_scan, n_garbled, total_chars, pdf_path)

    full_text = "\n\n".join(
        p["text"] for p in page_stats if p["type"] == PageContentType.TEXT.value and p["text"].strip()
    )
    if full_text:
        log_chunk_stats(full_text, chunk_size, chunk_overlap)

    if dump and full_text:
        logger.info("-" * 60)
        logger.info("Extracted text (first 2000 characters):")
        logger.info("%s", full_text[:DUMP_LIMIT])
        if len(full_text) > DUMP_LIMIT:
            logger.info("... (%d символов обрезано)", len(full_text) - DUMP_LIMIT)

    return {
        "file": str(pdf_path),
        "pages": total_pages,
        "n_text": n_text,
        "n_scan": n_scan,
        "n_garbled": n_garbled,
        "total_chars": total_chars,
    }


def ocr_hint(pdf_path: Path) -> None:
    logger.info("  Хорошая новость: ingestion.py теперь сам делает OCR сканов через PaddleOCR.")
    logger.info("    Убедись, что OCR_ENABLED=true")
    logger.info("    Просто запусти индексацию: python main.py ingest file '%s'", pdf_path)
    logger.info("  Если результат PaddleOCR неудовлетворителен — попробуй Surya:")
    logger.info("    pip install surya-ocr")
    logger.info("    OCR_ENGINE=auto python main.py ingest file '%s'", pdf_path)


def convert_hint(pdf_path: Path) -> None:
    logger.info("  Конвертировать через Ghostscript:")
    logger.info("    sudo apt install ghostscript")
    logger.info(
        '    gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -sOutputFile="%s_fixed.pdf" "%s"', pdf_path.stem, pdf_path
    )
    logger.info("  Или через LibreOffice:")
    logger.info('    libreoffice --headless --convert-to pdf "%s" --outdir .', pdf_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_pdf(args.pdf, dump=args.dump, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
