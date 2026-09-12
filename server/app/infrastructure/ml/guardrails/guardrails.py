"""PII detection and redaction for LLM output guardrails.

Scans text for PII patterns (phone numbers, emails, national IDs) across
Russian and Belarusian documents.  Zero external dependencies — stdlib ``re`` only.

Integrates with the dynamic config system via ``settings.pii_redaction_enabled``.
"""

from __future__ import annotations

import logging

from domain.value_objects.pii_patterns import ALL_PATTERNS

log = logging.getLogger("default")


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
