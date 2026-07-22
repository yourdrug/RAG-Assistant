"""Surya OCR adapter — infrastructure (optional)."""

from __future__ import annotations

import functools
import logging

from config import settings

log = logging.getLogger("default")


@functools.lru_cache(maxsize=1)
def _get_surya_predictors():
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor

    log.info("Loading Surya OCR ...")
    return (RecognitionPredictor(), DetectionPredictor())


def ocr_image_surya(image) -> str:
    rec_predictor, det_predictor = _get_surya_predictors()
    predictions = rec_predictor([image], [settings.ocr_lang_surya], det_predictor)
    lines = [line.text.strip() for line in predictions[0].text_lines if line.text.strip()]
    return "\n".join(lines)
