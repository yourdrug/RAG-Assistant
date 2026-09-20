"""PII redaction port — abstract interface for PII detection and redaction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PIIRedactorPort(Protocol):
    """Detect and redact PII in text. Handles enabled/disabled logic internally."""

    def redact(self, text: str) -> str: ...

    def scan_and_redact(self, text: str) -> tuple[str, list[str]]:
        """Scan for PII and return (redacted_text, list_of_detected_types)."""
        ...
