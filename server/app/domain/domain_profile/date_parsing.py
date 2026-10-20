"""Extract explicit temporal dates from questions, pending the RAG move."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from domain.shared.dates import parse_date_guess


if TYPE_CHECKING:
    from datetime import date


_QUESTION_AS_OF_RE = re.compile(
    r"(?:по состоянию на|на дату|на)\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})",
    re.IGNORECASE,
)


def explicit_question_date(text: str) -> date | None:
    """Extract a date explicitly requested as the temporal state of the answer."""
    match = _QUESTION_AS_OF_RE.search(text)
    if match is None:
        return None
    parsed = parse_date_guess(match.group("date"))
    if parsed is None:
        raise ValueError("Invalid date in question")
    return parsed
