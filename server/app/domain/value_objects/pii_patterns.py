"""PII patterns — compiled regex definitions for personally identifiable information.

These patterns encode the business rule "what constitutes PII" across
Russian and Belarusian documents.  Infrastructure code (guardrails)
uses these patterns for scanning/redaction but does not define them.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Universal PII patterns
# ---------------------------------------------------------------------------

# Phone: +7XXXXXXXXXX (RU), +375XXXXXXXXX (BY), 8XXXXXXXXXX, various separators
PHONE_RE = re.compile(r"(?<!\d)" r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}" r"(?!\d)")

PHONE_BY_RE = re.compile(r"(?<!\d)" r"\+375[\s\-]?\(?\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}" r"(?!\d)")

# Email
EMAIL_RE = re.compile(
    r"(?<![a-zA-Z0-9_.+-])" r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}" r"(?![a-zA-Z0-9_.+-])"
)

# Bank card number (16 digits, possibly with spaces/dashes)
CARD_RE = re.compile(r"(?<!\d)" r"\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}" r"(?!\d)")

# ---------------------------------------------------------------------------
# Russian document patterns
# ---------------------------------------------------------------------------

# Russian INN (10 or 12 digits)
RU_INN_RE = re.compile(r"(?<!\d)" r"(?:ИНН[:\s]?)?\d{10}(?:\d{2})?" r"(?!\d)")

# Russian SNILS (11 digits, formatted as XXX-XXX-XXX XX)
RU_SNILS_RE = re.compile(r"(?<!\d)" r"(?:СНИЛС[:\s]?)?\d{3}[\s\-]?\d{3}[\s\-]?\d{3}[\s]?\d{2}" r"(?!\d)")

# Russian passport series + number (4 digits + 6 digits)
RU_PASSPORT_RE = re.compile(r"(?<!\d)" r"(?:паспорт[:\s]?)?\d{4}\s?\d{6}" r"(?!\d)")

# Russian ОГРН (13 or 15 digits)
RU_OGRN_RE = re.compile(r"(?<!\d)" r"(?:ОГРН[:\s]?)?\d{13}(?:\d{2})?" r"(?!\d)")

# ---------------------------------------------------------------------------
# Belarusian document patterns
# ---------------------------------------------------------------------------

# Belarusian УНП (Учётный номер плательщика) — 9 digits
BY_UNP_RE = re.compile(r"(?<!\d)" r"(?:УНП[:\s]?)?\d{9}" r"(?!\d)")

# Belarusian passport: 2 letters + 7 digits (e.g. AB1234567)
BY_PASSPORT_RE = re.compile(r"(?<![A-Za-zА-Яа-яЁё])" r"[A-ZА-ЯЁ]{2}\d{7}" r"(?![A-Za-zА-Яа-яЁё\d])")

# Belarusian ID card number (14 digits)
BY_ID_CARD_RE = re.compile(r"(?<!\d)" r"(?:ID[-\s]?карт[ауы]?[:\s]?)?\d{14}" r"(?!\d)")


# All patterns: universal + country-specific
ALL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("phone", PHONE_RE),
    ("phone_by", PHONE_BY_RE),
    ("email", EMAIL_RE),
    ("card", CARD_RE),
    ("inn", RU_INN_RE),
    ("snils", RU_SNILS_RE),
    ("passport_ru", RU_PASSPORT_RE),
    ("ogrn", RU_OGRN_RE),
    ("unp_by", BY_UNP_RE),
    ("passport_by", BY_PASSPORT_RE),
    ("id_card_by", BY_ID_CARD_RE),
]
