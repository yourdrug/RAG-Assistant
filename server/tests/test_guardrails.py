"""Tests for PII detection and redaction (Russian + Belarusian patterns)."""

from infrastructure.ml.guardrails.guardrails import PIIDetector


class TestPIIDetector:
    def setup_method(self) -> None:
        self.detector = PIIDetector()

    def test_no_pii_returns_unchanged(self) -> None:
        text = "Обычный текст без персональных данных"
        result, found = self.detector.scan_and_redact(text)
        assert result == text
        assert found == []

    def test_phone_detection_ru(self) -> None:
        text = "Позвоните на +7 495 123 45 67 или 8-800-555-35-35"
        result, found = self.detector.scan_and_redact(text)
        assert "phone" in found
        assert "+7" not in result
        assert "8-800" not in result
        assert "***" in result

    def test_phone_detection_by(self) -> None:
        text = "Контакт: +375 29 123 45 67"
        result, found = self.detector.scan_and_redact(text)
        assert "phone_by" in found
        assert "+375" not in result

    def test_email_detection(self) -> None:
        text = "Напишите на user@example.com для связи"
        result, found = self.detector.scan_and_redact(text)
        assert "email" in found
        assert "user@example.com" not in result
        assert "***" in result

    def test_card_detection(self) -> None:
        text = "Карта 1234 5678 9012 3456"
        result, found = self.detector.scan_and_redact(text)
        assert "card" in found
        assert "1234 5678" not in result

    def test_inn_detection(self) -> None:
        text = "ИНН 1234567890 или ИНН 123456789012"
        result, found = self.detector.scan_and_redact(text)
        assert "inn" in found
        assert "1234567890" not in result

    def test_snils_detection(self) -> None:
        text = "СНИЛС 123-456-789 00"
        result, found = self.detector.scan_and_redact(text)
        assert "snils" in found
        assert "123-456" not in result

    def test_passport_detection_ru(self) -> None:
        text = "Паспорт 1234 567890"
        result, found = self.detector.scan_and_redact(text)
        assert "passport_ru" in found
        assert "1234 567890" not in result

    def test_ogrn_detection(self) -> None:
        text = "ОГРН 1234567890123"
        result, found = self.detector.scan_and_redact(text)
        assert "ogrn" in found
        assert "1234567890123" not in result

    def test_unp_detection_by(self) -> None:
        text = "УНП 123456789"
        result, found = self.detector.scan_and_redact(text)
        assert "unp_by" in found
        assert "123456789" not in result

    def test_passport_detection_by(self) -> None:
        text = "Пашпарт AB1234567"
        result, found = self.detector.scan_and_redact(text)
        assert "passport_by" in found
        assert "AB1234567" not in result

    def test_id_card_detection_by(self) -> None:
        text = "ID-карт 12345678901234"
        result, found = self.detector.scan_and_redact(text)
        assert "id_card_by" in found
        assert "12345678901234" not in result

    def test_multiple_pii_types(self) -> None:
        text = "Контакты: +7 495 123 45 67, email user@test.com, ИНН 1234567890"
        result, found = self.detector.scan_and_redact(text)
        assert len(found) >= 2
        assert "***" in result

    def test_empty_text(self) -> None:
        result, found = self.detector.scan_and_redact("")
        assert result == ""
        assert found == []

    def test_custom_mask(self) -> None:
        detector = PIIDetector(mask="[REDACTED]")
        text = "Позвоните на +7 495 123 45 67"
        result, _ = detector.scan_and_redact(text)
        assert "[REDACTED]" in result
