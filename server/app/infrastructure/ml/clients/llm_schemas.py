"""Pydantic schemas for structured LLM outputs.

Used with ``instructor`` to get type-safe, auto-retried responses from
LLM judge and relevance gate calls instead of fragile regex parsing.
"""

from __future__ import annotations

from typing import Protocol, cast

from pydantic import BaseModel, Field


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
) -> T:
    """Narrow the opaque provider handle at its structured-response boundary."""
    return await cast("StructuredClient", client).chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        response_model=response_model,
        max_retries=max_retries,
    )
