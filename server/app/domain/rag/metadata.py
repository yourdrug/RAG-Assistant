"""Non-secret metadata of an answer attempt, shared by sync and streaming."""

from dataclasses import dataclass

from domain.rag.outcome import AnswerOutcome


@dataclass(frozen=True, kw_only=True)
class AnswerMetadata:
    outcome: AnswerOutcome = AnswerOutcome.GENERATED
    request_id: str | None = None
    error_code: str | None = None
    candidates_count: int | None = None
    top_rerank_score: float | None = None
    condensed_query: str | None = None
    sub_queries: list[str] | None = None
    refined_query: str | None = None
    aux_tokens: int | None = None
    retrieval_ms: int | None = None
    rerank_ms: int | None = None
    config_hash: str | None = None
    config_version: str | None = None
    evidence_profile: str | None = None
    # Presentation the user asked for (AnswerStyle value) and why the LLM stopped generating.
    answer_style: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
