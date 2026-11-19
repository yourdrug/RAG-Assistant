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


# Results and traces recorded before answer styles stored the generation breadth instead.
LEGACY_BREADTH_STYLES: dict[Breadth, AnswerStyle] = {
    Breadth.NARROW: AnswerStyle.CONCISE,
    Breadth.BROAD: AnswerStyle.DETAILED,
}


def stored_answer_style(value: str) -> AnswerStyle:
    """An AnswerStyle value, or the legacy breadth label of an older result; anything else is an error."""
    if value in AnswerStyle:
        return AnswerStyle(value)

    if value in Breadth:
        return LEGACY_BREADTH_STYLES[Breadth(value)]

    message = f"Unknown answer style: {value!r}"
    raise ValueError(message)


def output_breadth(style: AnswerStyle) -> Breadth:
    """Output-limit class of a style: only a concise answer uses the smaller generation ceiling."""
    return Breadth.NARROW if style == AnswerStyle.CONCISE else Breadth.BROAD
