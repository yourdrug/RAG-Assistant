"""
pdf_diag.py — диагностика PDF файлов перед индексацией.

Что проверяет:
  1. Тип PDF: текстовый / сканированный / смешанный
  2. Количество извлечённого текста на страницу
  3. Качество текста (мусорные символы, кодировка)
  4. Итоговые чанки которые попадут в Qdrant
  5. Проблемные страницы

Запуск (из папки api/):
    pip install PyMuPDF
    python pdf_diag.py /path/to/file.pdf
    python pdf_diag.py /path/to/docs/          # все PDF в папке
    python pdf_diag.py /path/to/file.pdf --dump # показать полный текст
"""

import argparse
import sys
from pathlib import Path

# ── Цвета
G  = "\033[92m"
Y  = "\033[93m"
R  = "\033[91m"
C  = "\033[96m"
DIM = "\033[90m"
B  = "\033[1m"
RST = "\033[0m"


def is_garbled(text: str) -> bool:
    """Эвристика: если >15% символов — нечитаемый мусор, это скан без OCR."""
    if not text:
        return False
    total = len(text)
    # Считаем нормальные символы: буквы (включая кириллицу), цифры, пунктуация
    normal = sum(1 for c in text if c.isalnum() or c in ' .,;:!?-—\n\t()[]«»"\'')
    return (normal / total) < 0.6


def classify_page(text: str, chars: int) -> tuple:
    """
    Возвращает (тип, иконка, описание):
      text     — нормальный текстовый слой
      scan     — отсканированная страница без OCR (мало текста)
      garbled  — есть текст но нечитаемый (кривая кодировка / шрифт)
      empty    — пустая страница
    """
    if chars == 0:
        return "empty",   f"{DIM}○{RST}", "пустая"
    if chars < 50:
        return "scan",    f"{R}⊘{RST}", f"скан/изображение ({chars} симв)"
    if is_garbled(text):
        return "garbled", f"{Y}⚠{RST}", f"мусорный текст ({chars} симв)"
    return "text",       f"{G}✓{RST}", f"текст ({chars} симв)"


def check_pdf(pdf_path: Path, dump: bool = False, chunk_size: int = 512, chunk_overlap: int = 128):
    try:
        import fitz
    except ImportError:
        print(f"{R}PyMuPDF не установлен: pip install PyMuPDF{RST}")
        sys.exit(1)

    print(f"\n{B}{'─'*60}{RST}")
    print(f"{B}{C}{pdf_path.name}{RST}  ({pdf_path.stat().st_size / 1024:.0f} KB)")
    print(f"{'─'*60}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"Страниц: {total_pages}")

    # ── Анализ каждой страницы
    page_stats = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        chars = len(text.strip())
        ptype, icon, desc = classify_page(text, chars)
        page_stats.append({
            "num": i + 1,
            "type": ptype,
            "chars": chars,
            "text": text,
        })
        # Показываем только проблемные или первые 5
        if ptype != "text" or i < 3:
            print(f"  {icon} стр.{i+1:>3}: {desc}")
        elif i == 3 and all(p["type"] == "text" for p in page_stats):
            remaining_text = sum(1 for p in page_stats[3:] if p["type"] == "text")
            print(f"  {G}✓{RST} стр.4-{total_pages}: текст ({remaining_text + len(page_stats) - 3} страниц OK)")
            break

    doc.close()

    # ── Итоговая статистика
    types = [p["type"] for p in page_stats]
    n_text    = types.count("text")
    n_scan    = types.count("scan")
    n_garbled = types.count("garbled")
    n_empty   = types.count("empty")

    total_chars = sum(p["chars"] for p in page_stats)
    avg_chars   = total_chars // max(n_text, 1)

    print(f"\n{B}Итог:{RST}")
    print(f"  Текстовых:    {G}{n_text}{RST}/{total_pages}")
    if n_scan:    print(f"  Сканов:       {R}{n_scan}{RST}  ← нужен OCR")
    if n_garbled: print(f"  Мусорных:     {Y}{n_garbled}{RST}  ← проблема кодировки/шрифта")
    if n_empty:   print(f"  Пустых:       {DIM}{n_empty}{RST}")
    print(f"  Всего символов: {total_chars:,}")
    print(f"  Символов/стр (текст): ~{avg_chars:,}")

    # ── Диагноз
    print(f"\n{B}Диагноз:{RST}")

    if n_scan > n_text:
        print(f"  {R}✗ PDF содержит преимущественно сканы — текст НЕ извлечётся{RST}")
        print("    Решение: OCR через Tesseract (см. ниже)")
        ocr_hint(pdf_path)
    elif n_scan > 0:
        print(f"  {Y}~ PDF смешанный: {n_text} текстовых + {n_scan} сканов{RST}")
        print("    Текстовые страницы индексируются нормально.")
        print("    Для сканов нужен OCR.")
        ocr_hint(pdf_path)
    elif n_garbled > 0:
        print(f"  {Y}~ Мусорный текст на {n_garbled} стр. — проблема со шрифтами PDF{RST}")
        print("    Решение: конвертировать через LibreOffice или Ghostscript")
        convert_hint(pdf_path)
    elif total_chars < 500:
        print(f"  {R}✗ Слишком мало текста — документ скорее всего пустой или изображение{RST}")
    else:
        print(f"  {G}✓ PDF читается нормально, проблем не обнаружено{RST}")

    # ── Предпросмотр чанков
    full_text = "\n\n".join(p["text"] for p in page_stats if p["type"] == "text" and p["text"].strip())
    if full_text:
        chunks = simple_chunk(full_text, chunk_size, chunk_overlap)
        print(f"\n{B}Чанки (chunk_size={chunk_size}, overlap={chunk_overlap}):{RST}")
        print(f"  Итого чанков: {C}{len(chunks)}{RST}")
        if chunks:
            avg_chunk = sum(len(c) for c in chunks) / len(chunks)
            print(f"  Средний размер: {avg_chunk:.0f} символов")
            # Покажем первые 2 чанка
            for i, ch in enumerate(chunks[:2], 1):
                preview = ch[:120].replace('\n', '↵')
                print(f"  [{i}] {DIM}{preview}...{RST}")

    # ── Dump полного текста
    if dump and full_text:
        print(f"\n{B}{'─'*60}")
        print(f"ПОЛНЫЙ ТЕКСТ (первые 2000 символов):{RST}")
        print(full_text[:2000])
        if len(full_text) > 2000:
            print(f"\n{DIM}... ({len(full_text) - 2000} символов обрезано){RST}")

    return {
        "file": str(pdf_path),
        "pages": total_pages,
        "n_text": n_text,
        "n_scan": n_scan,
        "n_garbled": n_garbled,
        "total_chars": total_chars,
    }


def simple_chunk(text: str, size: int, overlap: int) -> list:
    """Упрощённый сплиттер для диагностики — тот же алгоритм что в ingestion.py."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def ocr_hint(pdf_path: Path):
    print(f"""
  {B}Хорошая новость: ingestion.py теперь сам делает OCR сканов через PaddleOCR.{RST}
    Убедись, что в config.py / переменных окружения:  OCR_ENABLED=true
    Просто запусти индексацию как обычно:
        python ingestion.py --file "{pdf_path}"

  {B}Если результат PaddleOCR неудовлетворителен — попробуй Surya:{RST}
    pip install surya-ocr   # отдельная лицензия весов, см. README → "Лицензии"
    OCR_ENGINE=auto python ingestion.py --file "{pdf_path}"
""")


def convert_hint(pdf_path: Path):
    print(f"""
  {B}Конвертировать через Ghostscript:{RST}
    sudo apt install ghostscript
    gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite \\
       -sOutputFile="{pdf_path.stem}_fixed.pdf" "{pdf_path}"

  {B}Или через LibreOffice:{RST}
    libreoffice --headless --convert-to pdf "{pdf_path}" --outdir .
""")


def main():
    parser = argparse.ArgumentParser(description="Диагностика PDF для RAG-индексации")
    parser.add_argument("path", help="Путь к PDF файлу или папке с PDF")
    parser.add_argument("--dump", action="store_true", help="Показать полный извлечённый текст")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    args = parser.parse_args()

    p = Path(args.path)

    if p.is_file():
        if p.suffix.lower() != ".pdf":
            print(f"{R}Файл не является PDF: {p}{RST}")
            sys.exit(1)
        check_pdf(p, dump=args.dump, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    elif p.is_dir():
        pdfs = list(p.glob("**/*.pdf")) + list(p.glob("**/*.PDF"))
        if not pdfs:
            print(f"{R}PDF файлов не найдено в {p}{RST}")
            sys.exit(1)

        print(f"{B}Найдено PDF: {len(pdfs)}{RST}")
        results = []
        for pdf in sorted(pdfs):
            r = check_pdf(pdf, dump=args.dump, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            results.append(r)

        # Общий итог по папке
        print(f"\n{B}{'═'*60}")
        print(f"СВОДКА ПО ПАПКЕ{RST}")
        print(f"{'═'*60}")
        total_ok    = sum(1 for r in results if r["n_scan"] == 0 and r["n_garbled"] == 0)
        total_scan  = sum(1 for r in results if r["n_scan"] > 0)
        total_garb  = sum(1 for r in results if r["n_garbled"] > 0)
        total_chars = sum(r["total_chars"] for r in results)
        print(f"  Читаются нормально: {G}{total_ok}{RST}/{len(results)}")
        if total_scan:  print(f"  Содержат сканы:     {R}{total_scan}{RST}  ← нужен OCR")
        if total_garb:  print(f"  Мусорный текст:     {Y}{total_garb}{RST}  ← нужна конвертация")
        print(f"  Итого символов:     {total_chars:,}")
        print()
    else:
        print(f"{R}Путь не найден: {p}{RST}")
        sys.exit(1)


if __name__ == "__main__":
    main()
