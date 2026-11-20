"""Pydantic schemas for structured LLM outputs.

Used with ``instructor`` to get type-safe, auto-retried responses from
LLM judge and relevance gate calls instead of fragile regex parsing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from config import LLMSection


OPENROUTER_HOST = "openrouter.ai"
# OpenRouter providers that accept the reasoning switch but keep thinking (checked 2026-11-19 on
# deepseek-v4.1-flash): routed there, a small output budget ends in reasoning with no content.
PROVIDERS_IGNORING_REASONING_OFF = ("novita",)


def reasoning_off_body(base_url: str) -> dict[str, Any]:
    """Request fields that switch off the thinking phase of hybrid reasoning models.

    OpenRouter normalizes reasoning control across providers and skips those known
    to ignore it; self-hosted servers (vLLM, SGLang) take the chat-template switch
    that Qwen3-style templates read.
    """
    host = (urlsplit(base_url).hostname or "").lower()

    if host == OPENROUTER_HOST or host.endswith(f".{OPENROUTER_HOST}"):
        return {
            "reasoning": {"enabled": False},
            "provider": {"ignore": list(PROVIDERS_IGNORING_REASONING_OFF)},
        }

    return {"chat_template_kwargs": {"enable_thinking": False}}


def auxiliary_request_body(llm: LLMSection) -> dict[str, Any] | None:
    """Extra request fields of auxiliary structured calls: reasoning off when configured.

    A hybrid reasoning model would otherwise spend the small output budget on
    thinking and return no content, which fails the structured response.
    """
    return reasoning_off_body(llm.llm_base_url) if llm.llm_disable_reasoning else None


class JudgeScore(BaseModel):
    """Structured response from the LLM judge."""

    score: float = Field(ge=0, le=10, description="Score from 0 to 10")
    reason: str = Field(description="One-sentence explanation of the score")


class RelevanceCheck(BaseModel):
    """Structured response from the relevance gate."""

    is_relevant: bool = Field(description="Whether the context is sufficient to answer the question")
    reason: str = Field(default="", description="Brief explanation if not relevant")
    suggested_refinement: str = Field(
        default="",
        description="If not relevant, a refined search query for one more retrieval attempt",
    )


class DecompositionCheck(BaseModel):
    """Structured response for whether a query needs decomposition."""

    needs_decomposition: bool = Field(description="Whether the query should be split into sub-queries")
    sub_queries: list[str] = Field(
        default_factory=list,
        description="List of sub-queries if decomposition is needed (2-4 items)",
    )


class StructuredCompletions(Protocol):
    async def create[T: BaseModel](
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        response_model: type[T],
        max_retries: int,
        extra_body: dict[str, Any] | None = None,
    ) -> T: ...


class StructuredChat(Protocol):
    @property
    def completions(self) -> StructuredCompletions: ...


class StructuredClient(Protocol):
    @property
    def chat(self) -> StructuredChat: ...


async def create_structured_response[T: BaseModel](
    client: object,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    response_model: type[T],
    max_retries: int,
    extra_body: dict[str, Any] | None = None,
) -> T:
    """Narrow the opaque provider handle at its structured-response boundary."""
    options: dict[str, Any] = {"extra_body": extra_body} if extra_body else {}
    return await cast("StructuredClient", client).chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        response_model=response_model,
        max_retries=max_retries,
        **options,
    )
