"""Tests for pure parsing/cleaning functions from domain/ingestion.py and infrastructure/registry.py.

No Qdrant/embeddings/OCR -- only text transformations, so they run fast without external services.
"""

from __future__ import annotations

from application.ports.file_storage import FileItem
from infrastructure.ml.ingestion import clean_pdf_text


def file_hash(source: FileItem) -> str:
    return f"{source.size_bytes}_{source.last_modified}"


def test_clean_pdf_text_dehyphenates_line_breaks() -> None:
    raw = "Компа-\nния заключила дого-\nвор"
    assert clean_pdf_text(raw) == "Компания заключила договор"


def test_clean_pdf_text_collapses_extra_whitespace() -> None:
    raw = "Строка   с      лишними    пробелами"
    assert clean_pdf_text(raw) == "Строка с лишними пробелами"


def test_clean_pdf_text_drops_decorative_lines() -> None:
    raw = "Заголовок\n------------\nТекст после разделителя\n•••"
    cleaned = clean_pdf_text(raw)
    assert "------------" not in cleaned
    assert "•••" not in cleaned
    assert "Заголовок" in cleaned
    assert "Текст после разделителя" in cleaned


def test_clean_pdf_text_drops_blank_lines_between_paragraphs() -> None:
    raw = "Абзац 1\n\n\n\n\nАбзац 2"
    # 5 newlines → collapsed to paragraph break (\n\n)
    assert clean_pdf_text(raw) == "Абзац 1\n\nАбзац 2"


def test_file_hash_changes_when_file_modified() -> None:

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


def test_file_hash_same_for_identical_items() -> None:

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
