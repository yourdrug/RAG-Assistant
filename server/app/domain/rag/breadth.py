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
