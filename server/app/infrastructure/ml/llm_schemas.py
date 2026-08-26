"""Pydantic schemas for structured LLM outputs.

Used with ``instructor`` to get type-safe, auto-retried responses from
LLM judge and relevance gate calls instead of fragile regex parsing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeScore(BaseModel):
    """Structured response from the LLM judge."""

    score: float = Field(ge=0, le=10, description="Score from 0 to 10")
    reason: str = Field(description="One-sentence explanation of the score")


class RelevanceCheck(BaseModel):
    """Structured response from the relevance gate."""

    is_relevant: bool = Field(description="Whether the context is sufficient to answer the question")
    reason: str = Field(default="", description="Brief explanation if not relevant")


class DecompositionCheck(BaseModel):
    """Structured response for whether a query needs decomposition."""

    needs_decomposition: bool = Field(description="Whether the query should be split into sub-queries")
    sub_queries: list[str] = Field(
        default_factory=list,
        description="List of sub-queries if decomposition is needed (2-4 items)",
    )


class SufficiencyAssessment(BaseModel):
    """Structured response for self-RAG sufficiency check."""

    is_sufficient: bool = Field(
        description="Whether the retrieved context is sufficient to answer the question"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why context is or isn't sufficient",
    )
    suggested_refinement: str = Field(
        default="",
        description="If insufficient, a refined query for broader retrieval",
    )
