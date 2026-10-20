"""RAG context-budget error, pending the RAG context move."""

from domain.shared.exceptions import ValidationError


class ContextBudgetExceededError(ValidationError):
    """No complete retrieved chunk and its conditions fit the context window."""

    code = "context_budget_exceeded"

    def __init__(self) -> None:
        super().__init__(
            "Полный контекст найденных документов не помещается в лимит ответа. "
            "Уточните вопрос или выберите конкретный пункт документа.",
            errors={"code": self.code},
        )
