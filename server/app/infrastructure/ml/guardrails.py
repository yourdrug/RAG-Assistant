"""PII detection and redaction for LLM output guardrails.

Scans text for PII patterns (phone numbers, emails, national IDs) across
Russian and Belarusian documents.  Zero external dependencies — stdlib ``re`` only.

Integrates with the dynamic config system via ``settings.pii_redaction_enabled``.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("default")

# ---------------------------------------------------------------------------
# Universal PII patterns
# ---------------------------------------------------------------------------

# Phone: +7XXXXXXXXXX (RU), +375XXXXXXXXX (BY), 8XXXXXXXXXX, various separators
_PHONE_RE = re.compile(
    r"(?<!\d)" r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}" r"(?!\d)"
)

_PHONE_BY_RE = re.compile(r"(?<!\d)" r"\+375[\s\-]?\(?\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}" r"(?!\d)")

# Email
_EMAIL_RE = re.compile(
    r"(?<![a-zA-Z0-9_.+-])" r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}" r"(?![a-zA-Z0-9_.+-])"
)

# Bank card number (16 digits, possibly with spaces/dashes)
_CARD_RE = re.compile(r"(?<!\d)" r"\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}" r"(?!\d)")

# ---------------------------------------------------------------------------
# Russian document patterns
# ---------------------------------------------------------------------------

# Russian INN (10 or 12 digits)
_RU_INN_RE = re.compile(r"(?<!\d)" r"(?:ИНН[:\s]?)?\d{10}(?:\d{2})?" r"(?!\d)")

# Russian SNILS (11 digits, formatted as XXX-XXX-XXX XX)
_RU_SNILS_RE = re.compile(r"(?<!\d)" r"(?:СНИЛС[:\s]?)?\d{3}[\s\-]?\d{3}[\s\-]?\d{3}[\s]?\d{2}" r"(?!\d)")

# Russian passport series + number (4 digits + 6 digits)
_RU_PASSPORT_RE = re.compile(r"(?<!\d)" r"(?:паспорт[:\s]?)?\d{4}\s?\d{6}" r"(?!\d)")

# Russian ОГРН (13 or 15 digits)
_RU_OGRN_RE = re.compile(r"(?<!\d)" r"(?:ОГРН[:\s]?)?\d{13}(?:\d{2})?" r"(?!\d)")

# ---------------------------------------------------------------------------
# Belarusian document patterns
# ---------------------------------------------------------------------------

# Belarusian УНП (Учётный номер плательщика) — 9 digits
_BY_UNP_RE = re.compile(r"(?<!\d)" r"(?:УНП[:\s]?)?\d{9}" r"(?!\d)")

# Belarusian passport: 2 letters + 7 digits (e.g. AB1234567)
_BY_PASSPORT_RE = re.compile(r"(?<![A-Za-zА-Яа-яЁё])" r"[A-ZА-ЯЁ]{2}\d{7}" r"(?![A-Za-zА-Яа-яЁё\d])")

# Belarusian ID card number (14 digits)
_BY_ID_CARD_RE = re.compile(r"(?<!\d)" r"(?:ID[-\s]?карт[ауы]?[:\s]?)?\d{14}" r"(?!\d)")

# Belarusian ОГРН (13 digits)
_BY_OGRN_RE = re.compile(r"(?<!\d)" r"(?:ОГРН[:\s]?)?\d{13}" r"(?!\d)")


# All patterns: universal + country-specific
ALL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("phone", _PHONE_RE),
    ("phone_by", _PHONE_BY_RE),
    ("email", _EMAIL_RE),
    ("card", _CARD_RE),
    ("inn", _RU_INN_RE),
    ("snils", _RU_SNILS_RE),
    ("passport_ru", _RU_PASSPORT_RE),
    ("ogrn", _RU_OGRN_RE),
    ("unp_by", _BY_UNP_RE),
    ("passport_by", _BY_PASSPORT_RE),
    ("id_card_by", _BY_ID_CARD_RE),
]


class PIIDetector:
    """Detect and redact PII in text.

    Usage::

        detector = PIIDetector()
        clean_text, found = detector.scan_and_redact(text)
        if found:
            log.warning("PII detected: %s", found)
    """

    _default: PIIDetector | None = None

    def __init__(self, *, mask: str = "***") -> None:
        self._mask = mask
        self._patterns = ALL_PATTERNS

    def scan(self, text: str) -> list[str]:
        """Scan text for PII patterns. Returns list of detected types."""
        found = []
        for pii_type, pattern in self._patterns:
            if pattern.search(text):
                found.append(pii_type)
        return found

    def scan_and_redact(self, text: str) -> tuple[str, list[str]]:
        """Scan for PII and return (redacted_text, list_of_detected_types).

        Redaction replaces matched PII with ``self.mask`` (default: "***").
        """
        found = self.scan(text)
        if not found:
            return text, found

        redacted = text
        for pii_type, pattern in self._patterns:
            if pii_type in found:
                redacted = pattern.sub(self._mask, redacted)

        log.info("PII redacted: types=%s, input_len=%d, output_len=%d", found, len(text), len(redacted))
        return redacted, found


def get_pii_detector() -> PIIDetector:
    """Get or create the default PII detector instance."""
    if PIIDetector._default is None:
        PIIDetector._default = PIIDetector()
    return PIIDetector._default


def invalidate_pii_detector() -> None:
    """Clear the cached detector instance (called on config change)."""
    PIIDetector._default = None
