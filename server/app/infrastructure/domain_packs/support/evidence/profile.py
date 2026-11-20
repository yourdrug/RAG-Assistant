"""Support evidence keeps product conditions and ordered actions attached to claims."""

from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile


READING_RULES = (
    "Проверь продукт, версию и условия применимости статьи перед ответом.",
    "Сохраняй порядок действий и предупреждения; не добавляй неподтверждённые шаги.",
    "Если решения нет в источниках, сообщи об этом и предложи обратиться в поддержку.",
)


class SupportEvidenceProfile(GenericEvidenceProfile):
    key = "support"

    def reading_rules(self, _question: str) -> list[str]:
        rules = "\n".join(f"- {rule}" for rule in READING_RULES)
        return [f"<support_article_reading>\n{rules}\n</support_article_reading>"]

    def reading_reminder(self, _question: str) -> str:
        rules = "\n".join(READING_RULES)
        return f"\n\n<answer_check>\nПри ответе по поддержке:\n{rules}\n</answer_check>"
