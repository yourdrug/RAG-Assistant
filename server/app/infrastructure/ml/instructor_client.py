"""Instructor-wrapped LLM clients for structured output.

Provides ``create_instructor_client`` that wraps OpenAI/Ollama clients
with ``instructor`` for Pydantic-validated, auto-retried LLM calls.
"""

from __future__ import annotations

import logging

import instructor
from openai import OpenAI

log = logging.getLogger("default")


def create_instructor_client(base_url: str, api_key: str = "ollama", model: str = ""):
    """Create an instructor-wrapped OpenAI client.

    Works with both Ollama (http://localhost:11434/v1) and OpenRouter.
    The returned client supports ``client.chat.completions.create()``
    with ``response_model=<Pydantic model>`` for structured output.
    """
    raw_client = OpenAI(base_url=base_url, api_key=api_key)
    return instructor.from_openai(raw_client)


def create_llm_instructor_client(model: str | None = None):
    """Create an instructor client configured for the active LLM provider.

    Returns (client, model_name). If *model* is given it is used as-is;
    otherwise the default model for the current provider is selected.
    """
    from config import settings
    from domain.value_objects.llm_provider import LLMProvider

    if settings.llm_provider == LLMProvider.OPENROUTER:
        client = create_instructor_client(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
        resolved_model = model or settings.openrouter_model
    else:
        client = create_instructor_client(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",
        )
        resolved_model = model or settings.llm_model
    return client, resolved_model
