"""Text cleaning utilities."""

from __future__ import annotations

import re

# A line that is *only* a page number / footer marker, e.g. "12", "- 12 -",
# "Page 12", "12/48", "стр. 12". These add noise to embeddings/search and
# rarely help retrieval, so we drop them outright.
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:[-–—]\s*)?"
    r"(?:(?:page|стр\.?|страница)\s*)?"
    r"\d{1,4}(?:\s*/\s*\d{1,4})?"
    r"(?:\s*[-–—])?\s*$",
    re.IGNORECASE,
)

# Smart quotes / dashes / non-breaking spaces -> plain equivalents. Keeps
# tokenization and exact-match search consistent regardless of how the
# source PDF encoded its typography.
_CHAR_NORMALIZE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\ufeff": "",
    }
)


def _normalize_chars(text: str) -> str:
    return text.translate(_CHAR_NORMALIZE_MAP)


def _strip_page_number_lines(text: str) -> str:
    lines = text.split("\n")
    kept = [ln for ln in lines if not _PAGE_NUMBER_RE.match(ln)]
    return "\n".join(kept)


def clean_pdf_text(text: str) -> str:
    r"""Clean extracted PDF text.

    - Rejoins genuinely hyphenated words split across a line break (only
      when both sides are word characters, so we don't accidentally eat
      list markers like "- item" or decorative dash runs).
    - Normalizes smart quotes/dashes/NBSP to plain ASCII equivalents.
    - Collapses runs of horizontal whitespace to a single space.
    - Removes decorative separator lines (---, ===, *** etc.) and
      page-number-only lines (running headers/footers).
    - Preserves paragraph breaks ("\\n\\n") instead of collapsing them to a
      single newline: downstream splitters (see splitting.py
      GENERAL_SEPARATORS) rely on "\\n\\n" as the primary chunk boundary, so
      destroying it here would make paragraph-aware splitting a no-op.
    """
    # Fix hyphenation at line breaks — only for mid-word breaks.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = _normalize_chars(text)
    # Collapse multiple horizontal whitespace to a single space (never touches \n).
    text = re.sub(r"[^\S\n]+", " ", text)
    # Remove decorative separator lines (---, ===, ***, box-drawing runs etc.)
    text = re.sub(r"^[•\-=~*_─━]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = _strip_page_number_lines(text)
    # Strip trailing whitespace left on each line after the removals above.
    text = re.sub(r"[ \t]+(?=\n)", "", text)
    # Collapse 3+ consecutive newlines down to a real paragraph break (2),
    # but keep single vs double newlines distinct.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
