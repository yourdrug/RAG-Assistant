"""Appendices of one source: separate roots classified on their own, only for splitting.

A resolution often carries an approved instruction, a contract carries a specification,
an instruction carries a form. The document keeps one ``doc_domain`` (ACL, versions and
answer policy depend on it); an appendix may only choose the profile that cuts its text.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from domain.ingestion.chunking_policy import ChunkingPolicy
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry


# Attribution block of an appendix: «Приложение 2\nк постановлению …», «Приложение\nк Инструкции …»,
# «УТВЕРЖДЕНО\nПостановление …». «Приложение: на 2 л.» and «УТВЕРЖДАЮ» in forms do not match.
APPENDIX_START_RE = re.compile(
    r"^[ \t]*(?:"
    r"Приложение(?:[ \t]*№?[ \t]*\d+(?:[-.]\d+)*)?[ \t]*\n?[ \t]*к[ \t]+\S"
    r"|УТВЕРЖДЕН[ОАЫ]?[ \t]*\n?[ \t]*(?:[Пп]остановлени|[Уу]каз|[Рр]ешени|[Пп]риказ|[Рр]аспоряжени)"
    r")",
    re.MULTILINE,
)
# A marker in the tail of a block usually only precedes the appendix title, which the
# parser starts as the next block. Cutting inside the block is worth it only when the
# appendix itself continues there (flat text, PDF without headings, single-block acts).
APPENDIX_MIN_TAIL_CHARS = 1000


# «Приложение 2\nк постановлению\nМинистерства …\n03.05.2021 N 17» without its own body.
MAX_ATTRIBUTION_CHARS = 400


def appendix_starts(text: str) -> list[int]:
    return [match.start() for match in APPENDIX_START_RE.finditer(text)]


def is_attribution_block(text: str) -> bool:
    """A block holding only the appendix attribution, not the appendix text."""
    stripped = text.strip()
    return len(stripped) <= MAX_ATTRIBUTION_CHARS and appendix_starts(stripped) == [0]


def appendix_cuts(
    texts: Sequence[str], *, min_tail_chars: int = APPENDIX_MIN_TAIL_CHARS
) -> list[tuple[int, int]]:
    """Return ordered ``(block index, offset)`` where each appendix root begins.

    The text before the first cut is the document body; a cut with nothing before it
    is not an appendix but the document itself.
    """
    cuts: list[tuple[int, int]] = []
    seen_text = False
    pending = False

    def add(index: int, offset: int) -> None:
        if seen_text and (not cuts or cuts[-1] != (index, offset)):
            cuts.append((index, offset))

    for index, text in enumerate(texts):
        if pending and text.strip():
            add(index, 0)
            pending = False

        for start in appendix_starts(text):
            if not text[:start].strip():
                add(index, 0)
                pending = False
            elif len(text) - start >= min_tail_chars:
                seen_text = True
                add(index, start)
                pending = False
            else:
                pending = True

        seen_text = seen_text or bool(text.strip())

    return cuts


def classify_appendix(
    registry: DomainProfileRegistry,
    text: str,
    *,
    document_domain: str,
    settings: DomainSettingsPort,
) -> str:
    """Domain whose profile splits the appendix; unclear signals inherit the document."""
    result = registry.classify(text, prior_domain=document_domain, settings=settings)

    if result.is_low_signal or not result.is_confident:
        return document_domain

    return result.domain_key


def appendix_policy(own: ChunkingPolicy, document: ChunkingPolicy) -> ChunkingPolicy:
    """Keep the document budgets; the appendix profile only decides boundaries and overlap.

    Budgets of another domain would re-cut wide form tables of a resolution and change
    chunk sizes within one document; a profile that forbids overlap still forbids it.
    """
    return replace(
        own,
        target_tokens=document.target_tokens,
        max_tokens=document.max_tokens,
        min_tokens=document.min_tokens,
        overlap_tokens=min(own.overlap_tokens, document.target_tokens - 1),
    )
