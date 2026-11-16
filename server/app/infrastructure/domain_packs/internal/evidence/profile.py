"""Internal answers distinguish documented duties from recommendations."""

from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile


class InternalEvidenceProfile(GenericEvidenceProfile):
    key = "internal"

    def reading_rules(self, _question: str) -> list[str]:
        return [
            "Проверь область применения внутреннего документа и упомянутые роли.",
            "Различай обязательные требования и рекомендации, сохраняй сроки и порядок согласования.",
            "При противоречиях укажи источники; не придумывай приоритет документа или исключения.",
        ]

    def reading_reminder(self, question: str) -> str:
        return "\n\nПри ответе по внутренним документам:\n" + "\n".join(self.reading_rules(question))
