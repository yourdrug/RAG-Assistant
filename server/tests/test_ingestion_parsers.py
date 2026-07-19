"""
Tests for pure parsing/cleaning functions from domain/ingestion.py and
infrastructure/registry.py.

No Qdrant/embeddings/OCR — only text transformations,
so they run fast without external services.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from domain.ingestion import clean_pdf_text  # noqa: E402
from infrastructure.registry import file_hash, is_already_indexed  # noqa: E402


def test_clean_pdf_text_dehyphenates_line_breaks():
    raw = "Компа-\nния заключила дого-\nвор"
    assert clean_pdf_text(raw) == "Компания заключила договор"


def test_clean_pdf_text_collapses_extra_whitespace():
    raw = "Строка   с      лишними    пробелами"
    assert clean_pdf_text(raw) == "Строка с лишними пробелами"


def test_clean_pdf_text_drops_decorative_lines():
    raw = "Заголовок\n------------\nТекст после разделителя\n•••"
    cleaned = clean_pdf_text(raw)
    assert "------------" not in cleaned
    assert "•••" not in cleaned
    assert "Заголовок" in cleaned
    assert "Текст после разделителя" in cleaned


def test_clean_pdf_text_drops_blank_lines_between_paragraphs():
    raw = "Абзац 1\n\n\n\n\nАбзац 2"
    assert clean_pdf_text(raw) == "Абзац 1\nАбзац 2"


def test_file_hash_changes_when_file_modified(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("версия 1", encoding="utf-8")
    h1 = file_hash(f)

    f.write_text("совсем другая по размеру версия текста", encoding="utf-8")
    h2 = file_hash(f)

    assert h1 != h2


def test_is_already_indexed_true_when_hash_matches(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("контент", encoding="utf-8")
    registry = {"doc.txt": {"hash": file_hash(f)}}
    assert is_already_indexed(f, registry) is True


def test_is_already_indexed_false_when_hash_differs(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("контент", encoding="utf-8")
    registry = {"doc.txt": {"hash": "0_0"}}
    assert is_already_indexed(f, registry) is False


def test_is_already_indexed_false_when_not_in_registry(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("контент", encoding="utf-8")
    assert is_already_indexed(f, {}) is False
