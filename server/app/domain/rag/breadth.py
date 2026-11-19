"""Question breadth modes and API aliases."""

from enum import StrEnum


class Breadth(StrEnum):
    """Question breadth modes for RAG pipeline."""

    NARROW = "narrow"
    BROAD = "broad"


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
