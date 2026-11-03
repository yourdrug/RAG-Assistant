"""OCR constructors stay lazy, cached and independently reloadable."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from application.runtime.config import ConfigParameterChanged, ConfigValueType
from config import settings
from infrastructure.ml.ingestion.ocr import (
    get_paddle_ocr,
    get_surya_predictors,
    invalidate_paddle_ocr_cache,
)


def test_paddle_constructor_is_cached_and_language_change_reloads(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = MagicMock()
    module = ModuleType("paddleocr")
    monkeypatch.setattr(module, "PaddleOCR", constructor, raising=False)
    monkeypatch.setattr(settings, "ocr_lang_paddle", "en")
    get_paddle_ocr.cache_clear()
    try:
        with patch.dict(sys.modules, {"paddleocr": module}):
            sys.modules.pop("infrastructure.ml.ingestion.paddle_backend", None)
            first = get_paddle_ocr()
            assert get_paddle_ocr() is first
            constructor.assert_called_once_with(use_angle_cls=True, lang="en", show_log=False)
            invalidate_paddle_ocr_cache(
                ConfigParameterChanged(
                    key="unrelated", old_value="en", new_value="ru", value_type=ConfigValueType.STR
                )
            )
            assert get_paddle_ocr() is first
            monkeypatch.setattr(settings, "ocr_lang_paddle", "ru")
            invalidate_paddle_ocr_cache(
                ConfigParameterChanged(
                    key="ocr_lang_paddle", old_value="en", new_value="ru", value_type=ConfigValueType.STR
                )
            )
            get_paddle_ocr()
            assert constructor.call_count == 2
            constructor.assert_called_with(use_angle_cls=True, lang="ru", show_log=False)
    finally:
        get_paddle_ocr.cache_clear()


def test_surya_loads_each_predictor_once(monkeypatch: pytest.MonkeyPatch) -> None:
    recognition, detection = MagicMock(), MagicMock()
    recognition_module, detection_module = ModuleType("surya.recognition"), ModuleType("surya.detection")
    monkeypatch.setattr(recognition_module, "RecognitionPredictor", recognition, raising=False)
    monkeypatch.setattr(detection_module, "DetectionPredictor", detection, raising=False)
    get_surya_predictors.cache_clear()
    try:
        with patch.dict(
            sys.modules, {"surya.recognition": recognition_module, "surya.detection": detection_module}
        ):
            sys.modules.pop("infrastructure.ml.ingestion.surya_backend", None)
            result = get_surya_predictors()
            assert result == (recognition.return_value, detection.return_value)
            assert get_surya_predictors() is result
            recognition.assert_called_once_with()
            detection.assert_called_once_with()
    finally:
        get_surya_predictors.cache_clear()
