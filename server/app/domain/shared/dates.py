"""Russian date parsing for effective-date extraction.

python-dateutil only understands English month names, so phrases like
Russian dates with year suffixes silently fail. This module parses Russian month
genitive/nominative forms explicitly, together with complete numeric and ISO dates.
"""

from __future__ import annotations

from datetime import date
import re


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

_RU_DATE_RE = re.compile(r"(\d{1,2})\s+([\u0430-яё]+)\s+(\d{4})", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")


def date_parts(text: str) -> tuple[int, int, int] | None:
    if m := _ISO_DATE_RE.search(text):
        return int(m[1]), int(m[2]), int(m[3])

    if m := _NUMERIC_DATE_RE.search(text):
        return int(m[3]), int(m[2]), int(m[1])

    if (m := _RU_DATE_RE.search(text)) and (month := _RU_MONTHS.get(m[2].lower())) is not None:
        return int(m[3]), month, int(m[1])

    return None


def parse_date_guess(text: str) -> date | None:
    """Parse a complete Russian, ISO, or numeric date without filling missing parts."""
    parts = date_parts(text) if text else None

    if parts is None:
        return None

    try:
        return date(*parts)
    except ValueError:
        return None
