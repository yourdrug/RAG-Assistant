"""
conftest.py — Mock heavy optional dependencies that aren't installed in dev/test.
surya-ocr is optional (requires separate install), so we mock it at import time.
"""

import sys
from unittest.mock import MagicMock

# Mock surya before any test module imports domain.ingestion
_surya_mock = MagicMock()
sys.modules.setdefault("surya", _surya_mock)
sys.modules.setdefault("surya.detection", _surya_mock)
sys.modules.setdefault("surya.recognition", _surya_mock)
