"""
pdf_diag.py — диагностика PDF файлов перед индексацией.

Что проверяет:
  1. Тип PDF: текстовый / сканированный / смешанный
  2. Количество извлечённого текста на страницу
  3. Качество текста (мусорные символы, кодировка)
  4. Итоговые чанки которые попадут в Qdrant
  5. Проблемные страницы
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz

logger = logging.getLogger("default")


def is_garbled(text: str) -> bool:
    """Эвристика: если >15% символов — нечитаемый мусор, это скан без OCR."""
    if not text:
        return False
    total = len(text)
    normal = sum(1 for c in text if c.isalnum() or c in " .,;:!?-—\n\t()[]«»\"'")
    return (normal / total) < 0.6


def classify_page(text: str, chars: int) -> tuple[str, str]:
    """
    Возвращает (тип, описание):
      text     — нормальный текстовый слой
      scan     — отсканированная страница без OCR (мало текста)
      garbled  — есть текст но нечитаемый (кривая кодировка / шрифт)
      empty    — пустая страница
    """
    if chars == 0:
        return "empty", "пустая"
    if chars < 50:
        return "scan", f"скан/изображение ({chars} симв)"
    if is_garbled(text):
        return "garbled", f"мусорный текст ({chars} симв)"
    return "text", f"текст ({chars} симв)"


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
        ptype, desc = classify_page(text, chars)
        page_stats.append(
            {
                "num": i + 1,
                "type": ptype,
                "chars": chars,
                "text": text,
            }
        )
        if ptype != "text" or i < 3:
            logger.info("  стр.%d: %s", i + 1, desc)
        elif i == 3 and all(p["type"] == "text" for p in page_stats):
            remaining_text = sum(1 for p in page_stats[3:] if p["type"] == "text")
            logger.info(
                "  стр.4-%d: текст (%d страниц OK)", total_pages, remaining_text + len(page_stats) - 3
            )
            break

    doc.close()

    types = [p["type"] for p in page_stats]
    n_text = types.count("text")
    n_scan = types.count("scan")
    n_garbled = types.count("garbled")
    n_empty = types.count("empty")

    total_chars = sum(p["chars"] for p in page_stats)
    avg_chars = total_chars // max(n_text, 1)

    logger.info("")
    logger.info("Итог:")
    logger.info("  Текстовых:    %d/%d", n_text, total_pages)
    if n_scan:
        logger.warning("  Сканов:       %d  ← нужен OCR", n_scan)
    if n_garbled:
        logger.warning("  Мусорных:     %d  ← проблема кодировки/шрифта", n_garbled)
    if n_empty:
        logger.info("  Пустых:       %d", n_empty)
    logger.info("  Всего символов: %s", f"{total_chars:,}")
    logger.info("  Символов/стр (текст): ~%s", f"{avg_chars:,}")

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

    full_text = "\n\n".join(p["text"] for p in page_stats if p["type"] == "text" and p["text"].strip())
    if full_text:
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
