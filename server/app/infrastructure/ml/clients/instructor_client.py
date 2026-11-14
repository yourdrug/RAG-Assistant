"""Instructor-wrapped LLM clients for structured output.

Provides ``create_instructor_client`` (sync) and ``create_async_instructor_client``
that wraps OpenAI-compatible clients with ``instructor`` for Pydantic-validated,
auto-retried LLM calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
import instructor
from openai import AsyncOpenAI, OpenAI

from config import LLMSection, process_section
from infrastructure.budget.transport import PaidAsyncTransport, PaidSyncTransport
from infrastructure.ml.usage_capture import record_model_response


if TYPE_CHECKING:
    from instructor import AsyncInstructor, Instructor
    from pydantic import SecretStr

    from infrastructure.budget.runtime import BudgetRuntime

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


def create_instructor_client(
    base_url: str, api_key: str, model: str = "", *, runtime: BudgetRuntime | None = None
) -> Instructor:
    """Create a sync instructor-wrapped OpenAI client.

    Uses the active LLM provider endpoint.
    The returned client supports ``client.chat.completions.create()``
    with ``response_model=<Pydantic model>`` for structured output.
    """
    del model
    raw_client = OpenAI(
        max_retries=0,
        base_url=base_url,
        api_key=api_key or NO_API_KEY,
        http_client=httpx.Client(
            follow_redirects=False,
            transport=PaidSyncTransport(runtime=runtime) if "openrouter.ai" in base_url else None,
        ),
    )
    client = instructor.from_openai(raw_client, mode=STRUCTURED_OUTPUT_MODE)
    client.on("completion:response", record_model_response)
    return client


def create_async_instructor_client(
    base_url: str, api_key: str, model: str = "", *, runtime: BudgetRuntime | None = None
) -> AsyncInstructor:
    """Create an async instructor-wrapped OpenAI client.

    Cached in MLClientRegistry to avoid creating new TCP connections per call.
    """
    del model
    raw_client = AsyncOpenAI(
        max_retries=0,
        base_url=base_url,
        api_key=api_key or NO_API_KEY,
        http_client=httpx.AsyncClient(
            follow_redirects=False,
            transport=PaidAsyncTransport(runtime=runtime) if "openrouter.ai" in base_url else None,
        ),
    )
    client = instructor.from_openai(raw_client, mode=STRUCTURED_OUTPUT_MODE)
    client.on("completion:response", record_model_response)
    return client


def create_llm_instructor_client(
    model: str | None = None, *, runtime: BudgetRuntime | None = None
) -> tuple[Instructor, str]:
    """Create a sync instructor client configured for the active LLM provider.

    Returns (client, model_name). If *model* is given it is used as-is;
    otherwise the default model for the current provider is selected.
    """
    llm_settings = process_section(LLMSection)

    client = create_instructor_client(
        runtime=runtime,
        base_url=llm_settings.active_llm_base_url,
        api_key=llm_settings.active_llm_api_key.get_secret_value(),
    )
    resolved_model = model or llm_settings.active_llm_model

    return client, resolved_model
