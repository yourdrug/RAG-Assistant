"""PII redactor adapter — wraps PIIDetector behind the application port."""

from __future__ import annotations

from config import settings


class PIIRedactorAdapter:
    """Implements PIIRedactorPort using infrastructure.ml.guardrails.PIIDetector."""

    def redact(self, text: str) -> str:
        if not settings.pii_redaction_enabled:
            return text

        from infrastructure.ml.guardrails import get_pii_detector

        detector = get_pii_detector()
        found = detector.scan(text)
        if not found:
            return text
        redacted, _ = detector.scan_and_redact(text)
        return redacted
