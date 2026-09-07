"""Text cleaning utilities."""

from __future__ import annotations

import re


def clean_pdf_text(text: str) -> str:
    """Clean extracted PDF text: fix hyphenation, collapse whitespace, remove decorative lines."""
    # Fix hyphenation at line breaks
    text = re.sub(r"-\n", "", text)
    # Collapse multiple whitespace to single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove decorative separator lines (---, ===, *** etc.)
    text = re.sub(r"^[•\-=~*]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()
