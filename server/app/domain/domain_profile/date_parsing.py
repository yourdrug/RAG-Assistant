"""Russian date parsing for effective-date extraction.

python-dateutil only understands English month names, so phrases like
«1 января 2026 г.» silently fail. This module parses Russian month
genitive/nominative forms explicitly and falls back to dateutil for
numeric formats (01.02.2026, 2026-02-01, ISO datetime).
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


def parse_date_guess(text: str) -> date | None:
    """Parse a date from a Russian or numeric text fragment. None on failure."""
    if not text:
        return None
    m = _RU_DATE_RE.search(text)
    if m:
        month = _RU_MONTHS.get(m.group(2).lower())
        if month is not None:
            try:
                return date(int(m.group(3)), month, int(m.group(1)))
            except ValueError:
                return None
    try:
        from dateutil import parser as dtparser

        return dtparser.parse(text, dayfirst=True).date()
    except Exception:
        return None
