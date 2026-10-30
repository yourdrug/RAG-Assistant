"""Profile-owned answer policy and temporal citation requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import date


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
