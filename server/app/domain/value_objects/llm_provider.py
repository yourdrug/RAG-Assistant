"""LLM provider constants.

Centralizes all magic strings used for provider selection
to avoid typos and enable IDE autocompletion.
"""

from enum import StrEnum


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
