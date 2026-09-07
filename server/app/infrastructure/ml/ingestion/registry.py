"""Parser registry — maps file extensions to parser functions."""

from __future__ import annotations

from infrastructure.ml.ingestion.docx import parse_docx
from infrastructure.ml.ingestion.markdown import parse_markdown
from infrastructure.ml.ingestion.rtf import parse_rtf
from infrastructure.ml.ingestion.txt import parse_txt

PARSERS = {
    ".docx": parse_docx,
    ".doc": parse_docx,
    ".rtf": parse_rtf,
    ".md": parse_markdown,
    ".txt": parse_txt,
}
