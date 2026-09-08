"""Text quality classification — pure domain functions with no side effects.

Shared by PDF, RTF, and DOCX preview strategies. Table detection
and IMAGE_ONLY handling stay in the callers (format-specific).
"""

from __future__ import annotations

from collections import Counter

from domain.value_objects.page_content_type import PageContentType

# Characters that signal uniform replacement (encoding errors, OCR failures)
_UNIFORM_REPLACEMENT_CHARS = frozenset("?_#\ufffd\u0000\x00")


def _is_uniform_replacement(text: str) -> bool:
    """Detect text that is mostly a single repeated replacement character."""
    non_ws = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if len(non_ws) < 3:
        return False
    char, count = Counter(non_ws).most_common(1)[0]
    ratio = count / len(non_ws)
    return ratio > 0.4 and char in _UNIFORM_REPLACEMENT_CHARS


def is_garbled(text: str) -> bool:
    """Return True if text is garbled (scan without OCR, encoding errors).

    Heuristic: >40% of characters are non-alphanumeric garbage.
    """
    if not text:
        return False
    # Table content (pipes, dashes, numbers) is not garbled
    if "|" in text and "---" in text:
        return False
    if _is_uniform_replacement(text):
        return True
    total = len(text)
    normal = sum(
        1 for c in text
        if c.isalnum() or c in " .,;:!-—\n\t()[]«»\"'"
    )
    return (normal / total) < 0.6


def classify_content(
    text: str,
    chars: int,
    has_table: bool = False,
    scan_threshold: int = 50,
) -> tuple[PageContentType, str]:
    """Classify content and return (type, description).

    Unified classifier for PDF/RTF/DOCX. Callers pass has_table=True
    when their format-specific detection finds table structures.

    Set scan_threshold=0 to disable the SCAN classification (used by
    RTF/DOCX which don't have scanned-page detection).

    Types: TEXT, SCAN, GARBLED, EMPTY, TABLE.
    """
    if chars == 0:
        if has_table:
            return PageContentType.TABLE, "таблица (без текстового слоя)"
        return PageContentType.EMPTY, "пустая"
    if scan_threshold > 0 and chars < scan_threshold:
        if has_table:
            return PageContentType.TABLE, "таблица"
        return PageContentType.SCAN, f"скан/изображение ({chars} симв)"
    if is_garbled(text):
        if has_table:
            return PageContentType.TABLE, "таблица"
        return PageContentType.GARBLED, f"мусорный текст ({chars} симв)"
    return PageContentType.TEXT, f"текст ({chars} симв)"
