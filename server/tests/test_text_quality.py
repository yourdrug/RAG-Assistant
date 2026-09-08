"""Tests for domain/services/text_quality.py — classify_content and is_garbled.

These are the shared classification functions extracted from pdf_diag.py
and used by PDF, RTF, and DOCX preview strategies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from domain.services.text_quality import classify_content, is_garbled  # noqa: E402
from domain.value_objects.page_content_type import PageContentType  # noqa: E402


# ---------------------------------------------------------------------------
# is_garbled
# ---------------------------------------------------------------------------


class TestIsGarbled:
    def test_empty_string(self):
        assert is_garbled("") is False

    def test_normal_text(self):
        assert is_garbled("Это нормальный текст на русском языке.") is False

    def test_english_text(self):
        assert is_garbled("This is normal English text.") is False

    def test_garbled_high_ratio(self):
        # Mostly non-alphanumeric garbage
        assert is_garbled("???###___???") is True

    def test_uniform_replacement(self):
        # Single repeated replacement character
        assert is_garbled("????????????????") is True
        assert is_garbled("________________") is True

    def test_table_not_garbled(self):
        # Table content with pipes and dashes is not garbled
        assert is_garbled("| колонка1 | колонка2 |\n|----------|----------|") is False

    def test_short_text_not_garbled(self):
        # Very short text should not be classified as garbled
        assert is_garbled("abc") is False

    def test_mixed_content(self):
        # Mostly normal with some garbage — should not be garbled
        assert is_garbled("Нормальный текст с一点点 мусора") is False


# ---------------------------------------------------------------------------
# classify_content — basic classification
# ---------------------------------------------------------------------------


class TestClassifyContent:
    def test_empty_text(self):
        ptype, desc = classify_content("", 0)
        assert ptype == PageContentType.EMPTY
        assert "пустая" in desc

    def test_short_text_is_scan(self):
        # Under 50 chars with scan_threshold=50 (default)
        ptype, desc = classify_content("Короткий текст", 14)
        assert ptype == PageContentType.SCAN
        assert "скан" in desc

    def test_normal_text(self):
        ptype, desc = classify_content("Это достаточно длинный нормальный текст для классификации.", 55)
        assert ptype == PageContentType.TEXT
        assert "текст" in desc

    def test_garbled_text(self):
        # Need scan_threshold=0 to bypass SCAN classification for short garbled text
        ptype, desc = classify_content("???###___???###___", 18, scan_threshold=0)
        assert ptype == PageContentType.GARBLED
        assert "мусорный" in desc


# ---------------------------------------------------------------------------
# classify_content — table detection
# ---------------------------------------------------------------------------


class TestClassifyContentTable:
    def test_empty_with_table(self):
        ptype, desc = classify_content("", 0, has_table=True)
        assert ptype == PageContentType.TABLE
        assert "текстового слоя" in desc

    def test_short_with_table(self):
        ptype, desc = classify_content("данные", 6, has_table=True)
        assert ptype == PageContentType.TABLE

    def test_garbled_with_table(self):
        ptype, desc = classify_content("???###___", 9, has_table=True)
        assert ptype == PageContentType.TABLE


# ---------------------------------------------------------------------------
# classify_content — scan_threshold parameter
# ---------------------------------------------------------------------------


class TestClassifyContentScanThreshold:
    def test_scan_threshold_zero_disables_scan(self):
        # With scan_threshold=0, short text is TEXT, not SCAN
        ptype, desc = classify_content("Короткий", 8, scan_threshold=0)
        assert ptype == PageContentType.TEXT

    def test_scan_threshold_custom(self):
        # With scan_threshold=10, text under 10 chars is SCAN
        ptype, desc = classify_content("Текст", 5, scan_threshold=10)
        assert ptype == PageContentType.SCAN

    def test_scan_threshold_not_reached(self):
        # Text above threshold is TEXT
        ptype, desc = classify_content("Достаточно длинный текст", 24, scan_threshold=10)
        assert ptype == PageContentType.TEXT


# ---------------------------------------------------------------------------
# classify_content — return type consistency
# ---------------------------------------------------------------------------


class TestClassifyContentReturnType:
    def test_returns_page_content_type_enum(self):
        for text, chars in [("", 0), ("абвгде", 6), ("нормальный текст", 16)]:
            ptype, desc = classify_content(text, chars)
            assert isinstance(ptype, PageContentType)
            assert isinstance(desc, str)
