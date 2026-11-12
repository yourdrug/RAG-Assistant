"""Render the plain-text chunking fixtures into the binary formats ingestion parses.

Sources live in ``tests/fixtures/chunking/*.txt`` so they stay reviewable in diffs.
Markup: a line starting with ``# `` is a large-font heading (PDF only), a line equal
to ``\\f`` is a page break (PDF only), consecutive ``|`` lines form a table (DOCX).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any

from docx import Document as WordDocument
import fitz


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "chunking"
PAGE_BREAK = "\\f"
HEADING_MARK = "# "
PDF_BODY_SIZE = 11
PDF_HEADING_SIZE = 18
PDF_MARGIN = 56
PDF_LINE_CHARS = 85


@dataclass(frozen=True)
class ChunkingFixture:
    """One source document, its rendered format and the domain it is split with."""

    name: str
    extension: str
    domain: str

    @property
    def source(self) -> Path:
        return FIXTURES_DIR / f"{self.name}.txt"

    @property
    def case_id(self) -> str:
        return f"{self.name}{self.extension}-{self.domain}"

    def render(self, directory: Path) -> Path:
        path = directory / f"{self.name}{self.extension}"
        lines = self.source.read_text(encoding="utf-8").splitlines()

        if self.extension == ".docx":
            render_docx(lines, path)
        elif self.extension == ".pdf":
            render_pdf(lines, path)
        else:
            path.write_text("\n".join(line for line in lines if line != PAGE_BREAK), encoding="utf-8")

        return path


def source_text(fixture: ChunkingFixture) -> str:
    """Meaningful text of the fixture without markup, for coverage checks."""
    lines = fixture.source.read_text(encoding="utf-8").splitlines()
    return "\n".join(line.removeprefix(HEADING_MARK) for line in lines if line != PAGE_BREAK)


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_delimiter_row(line: str) -> bool:
    return set(line.replace("|", "").strip()) <= {"-", ":", " "}


def add_docx_table(document: Any, rows: list[list[str]]) -> None:
    table = document.add_table(rows=len(rows), cols=len(rows[0]))

    for row, values in zip(table.rows, rows, strict=True):
        for cell, value in zip(row.cells, values, strict=True):
            cell.text = value


def render_docx(lines: list[str], path: Path) -> None:
    document = WordDocument()
    rows: list[list[str]] = []

    for line in [*lines, ""]:
        if line.startswith("|"):
            rows.extend([] if is_delimiter_row(line) else [table_cells(line)])
            continue

        if rows:
            add_docx_table(document, rows)
            rows = []

        if line and line != PAGE_BREAK:
            document.add_paragraph(line.removeprefix(HEADING_MARK))

    document.save(str(path))


def render_pdf(lines: list[str], path: Path) -> None:
    """Hard line breaks are real PDF lines; headings use a larger font size."""
    font = fitz.Font("cjk")  # Bundled with PyMuPDF and covers Cyrillic.
    document = fitz.Document()
    page = None
    y = 0.0

    def new_page() -> Any:
        created = document.new_page()
        created.insert_font(fontname="F0", fontbuffer=font.buffer)
        return created

    for line in lines:
        if line == PAGE_BREAK or page is None:
            page = new_page()
            y = PDF_MARGIN

            if line == PAGE_BREAK:
                continue

        heading = line.startswith(HEADING_MARK)
        size = PDF_HEADING_SIZE if heading else PDF_BODY_SIZE
        wrapped = [line.removeprefix(HEADING_MARK)] if heading else textwrap.wrap(line, PDF_LINE_CHARS)

        for text in wrapped:
            if y + size * 1.4 > page.rect.height - PDF_MARGIN:
                page = new_page()
                y = PDF_MARGIN

            y += size * 1.4
            page.insert_text((PDF_MARGIN, y), text, fontname="F0", fontsize=size)

        y += size * 0.6 if heading else 0

    document.save(str(path))
    document.close()


CHUNKING_FIXTURES = (
    ChunkingFixture("law", ".docx", "legal"),
    ChunkingFixture("law", ".txt", "legal"),
    ChunkingFixture("decree", ".docx", "decree"),
    # Contracts have no own profile yet (plan 5.3): split with legal and general.
    ChunkingFixture("contract", ".docx", "legal"),
    ChunkingFixture("contract", ".docx", "general"),
    ChunkingFixture("instruction", ".docx", "general"),
    ChunkingFixture("pdf_headings", ".pdf", "general"),
    ChunkingFixture("pdf_hard_wraps", ".pdf", "general"),
    ChunkingFixture("pdf_hard_wraps", ".pdf", "decree"),
    ChunkingFixture("table", ".docx", "general"),
)
