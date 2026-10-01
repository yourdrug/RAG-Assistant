"""Russian date parsing for effective-date extraction.

python-dateutil only understands English month names, so phrases like
«1 января 2026 г.» silently fail. This module parses Russian month
genitive/nominative forms explicitly, together with complete numeric and ISO dates.
"""

from __future__ import annotations

import re
from datetime import date

_RU_MONTHS: dict[str, int] = {
    "января": 1,
    "январь": 1,
    "февраля": 2,
    "февраль": 2,
    "марта": 3,
    "март": 3,
    "апреля": 4,
    "апрель": 4,
    "мая": 5,
    "май": 5,
    "июня": 6,
    "июнь": 6,
    "июля": 7,
    "июль": 7,
    "августа": 8,
    "август": 8,
    "сентября": 9,
    "сентябрь": 9,
    "октября": 10,
    "октябрь": 10,
    "ноября": 11,
    "ноябрь": 11,
    "декабря": 12,
    "декабрь": 12,
}

_RU_DATE_RE = re.compile(r"(\d{1,2})\s+([а-яё]+)\s+(\d{4})", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
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


def parse_date_guess(text: str) -> date | None:
    """Parse a complete Russian, ISO, or numeric date without filling missing parts."""
    if not text:
        return None
    if m := _ISO_DATE_RE.search(text):
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    if m := _NUMERIC_DATE_RE.search(text):
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    m = _RU_DATE_RE.search(text)
    if m:
        month = _RU_MONTHS.get(m.group(2).lower())
        if month is not None:
            try:
                return date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                return None
    return None
