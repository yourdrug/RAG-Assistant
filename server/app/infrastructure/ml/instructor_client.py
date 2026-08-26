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
