"""LLM provider and breadth constants.

Centralizes all magic strings used for provider selection and breadth modes
to avoid typos and enable IDE autocompletion.
"""

from enum import StrEnum


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class Breadth(StrEnum):
    """Question breadth modes for RAG pipeline."""

    NARROW = "narrow"
    BROAD = "broad"


# Breadth aliases used in API/CLI (mapped to canonical values)
BREADTH_ALIASES: dict[str, Breadth] = {
    "short": Breadth.NARROW,
    "detailed": Breadth.BROAD,
}
