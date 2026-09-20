"""PII redactor adapter — wraps PIIDetector behind the application port."""

from __future__ import annotations


class PIIRedactorAdapter:
    """Implements PIIRedactorPort using infrastructure.ml.guardrails.PIIDetector."""

    def __init__(self, pii_redaction_enabled: bool = False) -> None:
        self._enabled = pii_redaction_enabled

    def redact(self, text: str) -> str:
        if not self._enabled:
            return text

        from infrastructure.ml.guardrails.guardrails import get_pii_detector

        detector = get_pii_detector()
        found = detector.scan(text)
        if not found:
            return text
        redacted, _ = detector.scan_and_redact(text)
        return redacted

    def scan_and_redact(self, text: str) -> tuple[str, list[str]]:
        if not self._enabled:
            return text, []

        from infrastructure.ml.guardrails.guardrails import get_pii_detector

        detector = get_pii_detector()
        return detector.scan_and_redact(text)
