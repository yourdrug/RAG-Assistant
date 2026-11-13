"""Profile-owned answer policy and temporal citation requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import date


@dataclass(frozen=True)
class AnswerPolicy:
    rules: str | None = None
    fallback_to_full_corpus: bool = False
    # Profile-language bullet appended for an "as of" question; ``{date}`` is the requested date.
    as_of_note: str | None = None

    def prompt_addendum(
        self,
        as_of_date: date | None = None,
    ) -> str | None:
        if self.rules is None:
            return None

        if as_of_date is None or self.as_of_note is None:
            return self.rules

        return self.rules + self.as_of_note.format(date=as_of_date)

    def retrieval_fallback_to_full_corpus(self) -> bool:
        return self.fallback_to_full_corpus
