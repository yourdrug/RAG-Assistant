"""Instructor-wrapped LLM clients for structured output.

Provides ``create_async_instructor_client``, which wraps OpenAI-compatible clients with
``instructor`` for Pydantic-validated, auto-retried LLM calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import instructor
from openai import AsyncOpenAI

from infrastructure.ml.replay import llm_async_http_client
from infrastructure.ml.usage_capture import record_model_response


if TYPE_CHECKING:
    from instructor import AsyncInstructor
    from pydantic import SecretStr


log = logging.getLogger("default")

# JSON mode needs only response_format support; forced tool calls are unreliable on self-hosted
# servers (vLLM 0.10 V0 engine returns raw <tool_call> text as arguments).
STRUCTURED_OUTPUT_MODE = instructor.Mode.JSON
# Keyless OpenAI-compatible servers (vLLM without --api-key, Ollama) accept any non-empty key.
NO_API_KEY = "EMPTY"


def llm_auth_headers(api_key: SecretStr) -> dict[str, str]:
    """Bearer header for the active LLM endpoint; keyless self-hosted servers get none."""
    secret = api_key.get_secret_value()
    return {"Authorization": f"Bearer {secret}"} if secret else {}


def create_async_instructor_client(base_url: str, api_key: str, model: str = "") -> AsyncInstructor:
    """Create an async instructor-wrapped OpenAI client.

    Cached in MLClientRegistry to avoid creating new TCP connections per call.
    """
    del model
    raw_client = AsyncOpenAI(
        max_retries=0,
        base_url=base_url,
        api_key=api_key or NO_API_KEY,
        http_client=llm_async_http_client(),
    )
    client = instructor.from_openai(raw_client, mode=STRUCTURED_OUTPUT_MODE)
    client.on("completion:response", record_model_response)
    return client
