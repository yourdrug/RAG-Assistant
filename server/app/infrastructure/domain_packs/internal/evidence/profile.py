"""Internal answers distinguish documented duties from recommendations."""

from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile


READING_RULES = (
    "Проверь область применения внутреннего документа и упомянутые роли.",
    "Различай обязательные требования и рекомендации, сохраняй сроки и порядок согласования.",
    "При противоречиях укажи источники; не придумывай приоритет документа или исключения.",
)


class InternalEvidenceProfile(GenericEvidenceProfile):
    key = "internal"

    def reading_rules(self, _question: str) -> list[str]:
        rules = "\n".join(f"- {rule}" for rule in READING_RULES)
        return [f"<internal_document_reading>\n{rules}\n</internal_document_reading>"]

    def reading_reminder(self, _question: str) -> str:
        rules = "\n".join(READING_RULES)
        return f"\n\n<answer_check>\nПри ответе по внутренним документам:\n{rules}\n</answer_check>"
