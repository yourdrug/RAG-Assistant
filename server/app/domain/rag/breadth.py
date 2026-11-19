"""Question breadth, answer styles and API aliases.

``Breadth`` is the question classifier's estimate (or the user's explicit request) and is kept
for analytics. ``AnswerStyle`` is what the user asked for and changes only how the answer is
presented; retrieval, the context budget and the required factual completeness never depend on it.
"""

from enum import StrEnum


class Breadth(StrEnum):
    """Question breadth modes for RAG pipeline."""

    NARROW = "narrow"
    BROAD = "broad"


class AnswerStyle(StrEnum):
    """Presentation of the answer: the chat "Кратко / Авто / Подробно" selector."""

    CONCISE = "concise"
    ADAPTIVE = "adaptive"
    DETAILED = "detailed"


BREADTH_ALIASES: dict[str, Breadth] = {
    "short": Breadth.NARROW,
    "detailed": Breadth.BROAD,
}


def requested_breadth(depth: str | None) -> Breadth | None:
    """Map an explicit user request ("narrow"/"broad" or a legacy alias) to a breadth; None means auto."""
    if depth is None:
        return None

    if depth in BREADTH_ALIASES:
        return BREADTH_ALIASES[depth]

    try:
        return Breadth(depth)
    except ValueError:
        return None


def answer_style(depth: str | None) -> AnswerStyle:
    """Explicit "narrow" is concise, explicit "broad" is detailed; no (or unknown) request adapts."""
    requested = requested_breadth(depth)

    if requested is None:
        return AnswerStyle.ADAPTIVE

    return AnswerStyle.CONCISE if requested == Breadth.NARROW else AnswerStyle.DETAILED


def output_breadth(style: AnswerStyle) -> Breadth:
    """Output-limit class of a style: only a concise answer uses the smaller generation ceiling."""
    return Breadth.NARROW if style == AnswerStyle.CONCISE else Breadth.BROAD
