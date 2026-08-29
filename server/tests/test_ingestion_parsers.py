"""Tests for pure parsing/cleaning functions from domain/ingestion.py and infrastructure/registry.py.

No Qdrant/embeddings/OCR -- only text transformations, so they run fast without external services.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.ml.ingestion import clean_pdf_text  # noqa: E402
from infrastructure.registry import file_hash  # noqa: E402


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


def test_file_hash_changes_when_file_modified():
    from infrastructure.storage import FileItem

    item1 = FileItem(
        key="doc.txt",
        filename="doc.txt",
        size_bytes=100,
        last_modified="1000",
        extension=".txt",
    )
    h1 = file_hash(item1)

    item2 = FileItem(
        key="doc.txt",
        filename="doc.txt",
        size_bytes=200,
        last_modified="2000",
        extension=".txt",
    )
    h2 = file_hash(item2)

    assert h1 != h2


def test_file_hash_same_for_identical_items():
    from infrastructure.storage import FileItem

    item1 = FileItem(
        key="doc.txt",
        filename="doc.txt",
        size_bytes=100,
        last_modified="1000",
        extension=".txt",
    )
    item2 = FileItem(
        key="doc.txt",
        filename="doc.txt",
        size_bytes=100,
        last_modified="1000",
        extension=".txt",
    )
    assert file_hash(item1) == file_hash(item2)
