"""
domain/ingestion.py — Pure parsers + text splitting. No Qdrant, no storage, no registry.
"""

import html
import logging
import re
from pathlib import Path

from config import settings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = setup_logger()


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

_paddle_ocr_instance = None
_surya_predictors = None


def _get_paddle_ocr():
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        from paddleocr import PaddleOCR

        log.info("Загружаю PaddleOCR (lang=%s) ...", settings.ocr_lang_paddle)
        _paddle_ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang=settings.ocr_lang_paddle,
            show_log=False,
        )
    return _paddle_ocr_instance


def _get_surya_predictors():
    global _surya_predictors
    if _surya_predictors is None:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        log.info("Загружаю Surya OCR ...")
        _surya_predictors = (RecognitionPredictor(), DetectionPredictor())
    return _surya_predictors


def _ocr_image_paddle(image) -> str:
    import numpy as np

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
    import fitz
    from PIL import Image

    results = {}
    zoom = settings.ocr_dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for idx, page_num in enumerate(page_nums, 1):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        log.info("  OCR [%d/%d]  %s  стр. %d ...", idx, len(page_nums), filename, page_num + 1)

        text = ""
        engine_used = None
        try:
            if settings.ocr_engine in ("paddleocr", "auto"):
                text = _ocr_image_paddle(image)
                engine_used = "paddleocr"
            if (not text.strip()) and settings.ocr_engine in ("surya", "auto"):
                text = _ocr_image_surya(image)
                engine_used = "surya"
        except ImportError as e:
            log.error(
                "  OCR-движок '%s' не установлен (%s). " "См. requirements.txt / README для установки.",
                settings.ocr_engine,
                e,
            )
        except Exception as e:
            log.error("  Ошибка OCR на стр. %d: %s", page_num + 1, e)

        if text.strip():
            results[page_num] = clean_pdf_text(text)
            log.info("  OCR [%d/%d]  OK (%s) — %d символов", idx, len(page_nums), engine_used, len(text))
        else:
            log.warning("  OCR [%d/%d]  пусто — страница %d пропущена", idx, len(page_nums), page_num + 1)

    return results


def clean_pdf_text(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        if line and not re.match(r"^[\s\-=_.|•·▪]+$", line):
            cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------


def parse_pdf(path: Path) -> list[Document]:
    try:
        import fitz

        doc = fitz.open(str(path))
        pages_text: dict[int, str] = {}
        scan_page_nums = []

        for i, page in enumerate(doc):
            blocks = page.get_text("blocks", sort=True)
            page_parts = [block[4].strip() for block in blocks if block[6] == 0 and block[4].strip()]
            page_text = clean_pdf_text("\n".join(page_parts))

            if len(page_text.strip()) < 50:
                scan_page_nums.append(i)
            else:
                pages_text[i] = page_text

        if scan_page_nums:
            pages_str = ", ".join(str(p + 1) for p in scan_page_nums[:10])
            suffix = f"... (+{len(scan_page_nums) - 10})" if len(scan_page_nums) > 10 else ""
            log.warning("  Страницы без текстового слоя (скан?): %s%s", pages_str, suffix)

            if settings.ocr_enabled:
                ocr_results = ocr_pdf_pages(doc, scan_page_nums, path.name)
                pages_text.update(ocr_results)
            else:
                log.warning("  OCR выключен (OCR_ENABLED=false) — страницы-сканы пропущены: %s", path.name)

        doc.close()

        if not pages_text:
            raise RuntimeError(
                "PDF состоит из сканов, и OCR не смог извлечь текст (или отключён). "
                "Проверь качество скана / включи OCR_ENABLED=true."
            )

        return [
            Document(
                page_content=pages_text[i],
                metadata={"page": i + 1},
            )
            for i in sorted(pages_text.keys())
        ]

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"PDF parse error: {e}")


# ---------------------------------------------------------------------------
# DOCX parser
# ---------------------------------------------------------------------------


def parse_docx(path: Path) -> str:
    try:
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        parts = []
        for para in doc.paragraphs:
            if t := para.text.strip():
                parts.append(t)
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts)
    except Exception as e:
        return _parse_docx_raw(path, original_error=str(e))


def _parse_docx_raw(path: Path, original_error: str = "") -> str:
    import xml.etree.ElementTree as ET
    import zipfile

    try:
        with zipfile.ZipFile(str(path), "r") as z:
            if "word/document.xml" not in z.namelist():
                raise RuntimeError("word/document.xml not found")
            with z.open("word/document.xml") as f:
                tree = ET.parse(f)
        ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        texts = [e.text for e in tree.getroot().iter(f"{{{ns}}}t") if e.text and e.text.strip()]
        result = " ".join(texts)
        if not result.strip():
            raise RuntimeError("Пустой текст после XML-парсинга")
        return result
    except Exception as e:
        raise RuntimeError(f"DOCX parse failed (python-docx: {original_error}, raw XML: {e})")


# ---------------------------------------------------------------------------
# RTF / MD / TXT
# ---------------------------------------------------------------------------


def parse_rtf(path: Path) -> str:
    try:
        from striprtf.striprtf import rtf_to_text

        raw = path.read_bytes()
        for enc in ("utf-8", "cp1251", "cp1252", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
        result = re.sub(r"\n{3,}", "\n\n", rtf_to_text(text)).strip()
        if not result:
            raise RuntimeError("Пустой текст после RTF-конвертации")
        return result
    except ImportError:
        raise RuntimeError("striprtf не установлен: pip install striprtf")
    except Exception as e:
        raise RuntimeError(f"RTF parse error: {e}")


def parse_md(path: Path) -> str:
    try:
        import markdown as md_lib

        raw = path.read_text(encoding="utf-8", errors="replace")
        html_content = md_lib.markdown(raw)
        text = re.sub(r"<[^>]+>", " ", html_content)
        return re.sub(r"\n{3,}", "\n\n", html.unescape(text)).strip()
    except ImportError:
        return path.read_text(encoding="utf-8", errors="replace")


def parse_txt(path: Path) -> str:
    for enc in ("utf-8", "cp1251", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Dispatcher + splitter
# ---------------------------------------------------------------------------

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".doc": parse_docx,
    ".rtf": parse_rtf,
    ".md": parse_md,
    ".txt": parse_txt,
}


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    log.info("Сплиттер: %d чанков из %d документов", len(chunks), len(docs))
    return chunks
