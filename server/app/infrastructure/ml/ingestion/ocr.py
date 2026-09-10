"""OCR functions — lazy-loaded PaddleOCR and Surya runners."""

from __future__ import annotations

import functools
import logging

import numpy as np
from PIL import Image

from config import settings

log = logging.getLogger("detailed")

# Recognitions below this confidence are dropped rather than fed into search
# — low-confidence OCR guesses are noise that actively hurts embeddings/exact
# match, not signal. Overridable via settings.ocr_min_confidence.
_DEFAULT_OCR_MIN_CONFIDENCE = 0.5


@functools.lru_cache(maxsize=1)
def get_paddle_ocr():
    from paddleocr import PaddleOCR

    log.info("Loading PaddleOCR (lang=%s) ...", settings.ocr_lang_paddle)
    return PaddleOCR(
        use_angle_cls=True,
        lang=settings.ocr_lang_paddle,
        show_log=False,
    )


@functools.lru_cache(maxsize=1)
def get_surya_predictors():
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor

    log.info("Loading Surya OCR ...")
    return (RecognitionPredictor(), DetectionPredictor())


def _bbox_from_points(box) -> tuple[float, float, float, float]:
    """Convert a 4-point quadrilateral (PaddleOCR box format) to (x0, y0, x1, y1)."""
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    return min(xs), min(ys), max(xs), max(ys)


def _order_ocr_entries(entries: list[dict], min_confidence: float | None = None) -> list[str]:
    """Order OCR-recognized text lines into reading order, dropping low-confidence noise.

    Each entry is {"text": str, "confidence": float | None, "bbox": (x0, y0, x1, y1)}.
    OCR engines don't guarantee their raw output order follows the page's
    reading order — this is especially visible on skewed scans or
    multi-column layouts, where lines can come back in a near-arbitrary
    sequence. We group entries into rows by y-position (using the median
    detected line height as the row tolerance, so it adapts to the actual
    font size instead of a fixed pixel threshold) and sort left-to-right
    within each row, then rows top-to-bottom.
    """
    threshold = _DEFAULT_OCR_MIN_CONFIDENCE if min_confidence is None else min_confidence
    kept = [e for e in entries if e["confidence"] is None or e["confidence"] >= threshold]
    if not kept:
        return []

    heights = sorted(e["bbox"][3] - e["bbox"][1] for e in kept if e["bbox"][3] > e["bbox"][1])
    median_height = heights[len(heights) // 2] if heights else 15.0
    row_tolerance = max(median_height * 0.6, 1.0)

    kept.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
    rows: list[list[dict]] = []
    for e in kept:
        y0 = e["bbox"][1]
        if rows:
            row_y = sum(r["bbox"][1] for r in rows[-1]) / len(rows[-1])
            if abs(y0 - row_y) <= row_tolerance:
                rows[-1].append(e)
                continue
        rows.append([e])

    lines = []
    for row in rows:
        row.sort(key=lambda e: e["bbox"][0])
        line_text = " ".join(e["text"] for e in row if e["text"])
        if line_text:
            lines.append(line_text)
    return lines


def ocr_image_paddle(image) -> str:
    ocr = get_paddle_ocr()
    result = ocr.ocr(np.array(image), cls=True)
    min_confidence = getattr(settings, "ocr_min_confidence", _DEFAULT_OCR_MIN_CONFIDENCE)

    entries = []
    for block in result or []:
        for detection in block or []:
            box, (text, confidence) = detection[0], detection[1]
            text = (text or "").strip()
            if not text:
                continue
            entries.append({"text": text, "confidence": confidence, "bbox": _bbox_from_points(box)})

    return "\n".join(_order_ocr_entries(entries, min_confidence))


def ocr_image_surya(image) -> str:
    rec_predictor, det_predictor = get_surya_predictors()
    predictions = rec_predictor([image], [settings.ocr_lang_surya], det_predictor)
    min_confidence = getattr(settings, "ocr_min_confidence", _DEFAULT_OCR_MIN_CONFIDENCE)

    entries = []
    for i, line in enumerate(predictions[0].text_lines):
        text = (line.text or "").strip()
        if not text:
            continue
        bbox = getattr(line, "bbox", None)
        if bbox is None or len(bbox) < 4:
            # No usable geometry — fall back to a synthetic strictly-increasing
            # bbox so entries without positions still preserve engine order
            # relative to each other instead of being dropped or collapsed
            # onto the same row.
            bbox = (0.0, float(i), 0.0, float(i) + 1.0)
        confidence = getattr(line, "confidence", None)
        entries.append({"text": text, "confidence": confidence, "bbox": tuple(bbox)})

    return "\n".join(_order_ocr_entries(entries, min_confidence))


def ocr_pdf_pages(doc, page_nums: list[int], filename: str) -> dict:
    import fitz

    results = {}
    zoom = settings.ocr_dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for page_num in page_nums:
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        text = ""
        if settings.ocr_engine in ("paddleocr", "auto"):
            text = ocr_image_paddle(img)

        if not text and settings.ocr_engine in ("surya", "auto"):
            text = ocr_image_surya(img)

        results[page_num] = text

    return results
