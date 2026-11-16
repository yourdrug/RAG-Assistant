"""Support evidence keeps product conditions and ordered actions attached to claims."""

from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile


class SupportEvidenceProfile(GenericEvidenceProfile):
    key = "support"

    def reading_rules(self, _question: str) -> list[str]:
        return [
            "Проверь продукт, версию и условия применимости статьи перед ответом.",
            "Сохраняй порядок действий и предупреждения; не добавляй неподтверждённые шаги.",
            "Если решения нет в источниках, сообщи об этом и предложи обратиться в поддержку.",
        ]

    def reading_reminder(self, question: str) -> str:
        return "\n\nПри ответе по поддержке:\n" + "\n".join(self.reading_rules(question))
