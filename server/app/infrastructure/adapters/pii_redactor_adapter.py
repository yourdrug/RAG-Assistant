"""PII redactor adapter — wraps PIIDetector behind the application port."""

from __future__ import annotations

from typing import TYPE_CHECKING

from infrastructure.ml.guardrails.guardrails import PIIDetector


if TYPE_CHECKING:
    from application.ports.rag_settings import RagSettingsPort


class PIIRedactorAdapter:
    """Implements PIIRedactorPort using infrastructure.ml.guardrails.PIIDetector."""

    def __init__(self, rag_settings: RagSettingsPort) -> None:
        self._settings = rag_settings
        self.detector = PIIDetector()

    def redact(self, text: str) -> str:
        redacted, _ = self.scan_and_redact(text)
        return redacted

    def scan_and_redact(self, text: str) -> tuple[str, list[str]]:
        if not self._settings.pii_redaction_enabled:
            return text, []

        return self.detector.scan_and_redact(text)
