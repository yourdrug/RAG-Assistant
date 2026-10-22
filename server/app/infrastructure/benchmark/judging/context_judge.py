"""Judge retrieved context separately from deterministic metrics."""

import logging
from infrastructure.benchmark.judging.prompts import CONTEXT_PRECISION_PROMPT, CONTEXT_RECALL_PROMPT

from config import settings
from domain.runtime.enums import LLMProvider
from langchain.schema import Document
from infrastructure.benchmark.judging import judge


logger = logging.getLogger("default")


def compute_context_precision_recall(
    question: str,
    answer: str,
    docs_with_scores: list[tuple[Document, float]] | list[dict] | None = None,
    judge_llm=None,
    context_override: str | None = None,
    *,
    judge_model: str | None = None,
    saved_metrics: dict | None = None,
    metric_callback=None,
) -> dict:
    """Compute context_precision and context_recall via LLM judge.

    Args:
        question: The user question.
        answer: The generated answer.
        docs_with_scores: Retrieved documents with similarity scores.
        judge_llm: Optional pre-configured judge LLM.
        context_override: If provided, use this as the context string directly
            instead of extracting from docs_with_scores.
        judge_model: Explicit judge model; falls back to the configured model when omitted.
        saved_metrics: Successful scores reused on resume.
        metric_callback: Persist each metric after its judge call.

    """
    if not context_override and not docs_with_scores:
        return {
            "context_precision": None,
            "context_precision_reason": "No documents retrieved",
            "context_recall": None,
            "context_recall_reason": "No documents retrieved",
        }

    model = judge_model or judge.get_judge_model()

    if context_override:
        context = context_override
    else:
        if docs_with_scores is None:
            raise ValueError("docs_with_scores must be provided when context_override is not set")

        context = "\n\n---\n\n".join(
            d.page_content if hasattr(d, "page_content") else str(d) for d, _ in docs_with_scores
        )

    result = dict(saved_metrics or {})

    for key, template in (
        ("context_precision", CONTEXT_PRECISION_PROMPT),
        ("context_recall", CONTEXT_RECALL_PROMPT),
    ):
        if result.get(key) is not None:
            continue

        try:
            client = judge.get_judge_client(model if settings.llm_provider == LLMProvider.OLLAMA else "")
            score = judge.judge_with_structured_output(
                client, template.format(question=question, answer=answer, context=context), model
            )
            result[key] = max(0.0, min(10.0, score.score))
            result[f"{key}_reason"] = score.reason
            result.pop(f"{key}_error", None)
        except Exception as exc:
            logger.warning("Context judge failed (%s): %s", key, exc)
            result[key] = None
            result[f"{key}_error"] = str(exc)
            result[f"{key}_reason"] = f"[Error: {exc}]"

        if metric_callback is not None:
            metric_callback(dict(result))

    return result
