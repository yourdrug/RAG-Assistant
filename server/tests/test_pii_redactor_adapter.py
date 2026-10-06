"""PII redaction follows live configuration on the same adapter instance."""

from unittest.mock import patch

import pytest

from config import settings
from infrastructure.adapters.pii_redactor_adapter import PIIRedactorAdapter
from infrastructure.ml.config.settings_adapters import LiveRagSettings


def test_redaction_tracks_enable_disable_and_reenable(monkeypatch):
    monkeypatch.setattr(settings, "pii_redaction_enabled", False)
    redactor = PIIRedactorAdapter(rag_settings=LiveRagSettings())
    text = "Контакт: private@example.com"

    for enabled in (False, True, False, True):
        monkeypatch.setattr(settings, "pii_redaction_enabled", enabled)
        expected = "Контакт: ***" if enabled else text
        assert redactor.redact(text) == expected
        assert redactor.scan_and_redact(text) == (expected, ["email"] if enabled else [])


@pytest.mark.parametrize("text", ["", "Текст без персональных данных"])
@pytest.mark.parametrize("enabled", [False, True])
def test_text_without_pii_is_unchanged(monkeypatch, text, enabled):
    monkeypatch.setattr(settings, "pii_redaction_enabled", enabled)
    redactor = PIIRedactorAdapter(rag_settings=LiveRagSettings())

    assert redactor.redact(text) == text
    assert redactor.scan_and_redact(text) == (text, [])


def test_disabled_redaction_does_not_load_detector(monkeypatch):
    monkeypatch.setattr(settings, "pii_redaction_enabled", False)
    redactor = PIIRedactorAdapter(rag_settings=LiveRagSettings())
    text = "private@example.com"

    with patch("infrastructure.ml.guardrails.guardrails.get_pii_detector") as detector:
        assert redactor.redact(text) == text
        assert redactor.scan_and_redact(text) == (text, [])
        detector.assert_not_called()
