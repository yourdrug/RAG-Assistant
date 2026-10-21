"""Domain-specific answer rules, independent of document ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.knowledge.enums import DocDomain


if TYPE_CHECKING:
    from datetime import date


_LEGAL_PROMPT_RULES = (
    "- Обязательно указывай номер статьи/пункта, если он есть в контексте "
    "(например: «Согласно ст. 15 ФЗ-XXX» или «п. 3.2 Договора»).\n"
    "- \u041d\u0435 перефразируй формулировки нормативных актов — цитируй максимально близко к тексту.\n"
)

_DECREE_PROMPT_RULES = (
    "- Каждый указ идентифицируй по номеру и дате.\n"
    "- Ссылайся на конкретный пункт постановляющей части, не пересказывай указ целиком.\n"
)


def versioned_prompt_date_stamp(as_of_date: date | None) -> str:
    """Generate the shared date-stamp bullet for versioned domain prompt addenda."""
    if as_of_date is None:
        return ""

    return (
        f"- Ответ дан по состоянию на {as_of_date.strftime('%d.%m.%Y')}. "
        "Если подходят разные редакции — укажи, какая редакция использована.\n"
    )


@dataclass(frozen=True)
class AnswerPolicy:
    rules: str | None = None
    fallback_to_full_corpus: bool = False

    def prompt_addendum(
        self,
        as_of_date: date | None = None,
    ) -> str | None:
        if self.rules is None:
            return None

        return self.rules + versioned_prompt_date_stamp(as_of_date)

    def retrieval_fallback_to_full_corpus(self) -> bool:
        return self.fallback_to_full_corpus


def policy_for(domain_key: str) -> AnswerPolicy:
    if domain_key == DocDomain.LEGAL:
        return AnswerPolicy(_LEGAL_PROMPT_RULES, fallback_to_full_corpus=True)

    if domain_key == DocDomain.DECREE:
        return AnswerPolicy(_DECREE_PROMPT_RULES, fallback_to_full_corpus=True)

    return AnswerPolicy()
