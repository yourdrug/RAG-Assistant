"""Benchmark metrics — retriever metrics, context precision/recall, summary."""

from application.benchmark.summary import summarize_benchmark_results
from domain.services.benchmark_judge import result_judge_errors  # noqa: F401

import logging
from pathlib import Path

from langchain.schema import Document

from infrastructure.benchmark.source_hints import matches_source_hints, parse_source_hints

logger = logging.getLogger("default")


def _extract_source_name(doc: Document) -> str:
    """Extract clean source name from document metadata."""
    filename = doc.metadata.get("filename", "")
    if filename:
        return filename
    source = doc.metadata.get("source", "")
    if source:
        return Path(source).name
    return "?"


def compute_retriever_metrics(
    docs_with_scores: list[tuple[Document, float]],
    source_hint: str | None,
) -> dict:
    scores_list = [s for _, s in docs_with_scores]
    avg_sim = sum(scores_list) / len(scores_list) if scores_list else 0.0

    hints = parse_source_hints(source_hint)
    if not hints:
        return {
            "hit_rate": None,
            "mrr": None,
            "avg_similarity": round(avg_sim, 4),
            "retrieved_sources": [_extract_source_name(d) for d, _ in docs_with_scores],
        }

    hit_rate = 0
    mrr = 0.0
    for rank, (doc, _) in enumerate(docs_with_scores, 1):
        filename = doc.metadata.get("filename", "") or doc.metadata.get("source", "")
        if matches_source_hints(filename, hints):
            hit_rate = 1
            if mrr == 0.0:
                mrr = 1.0 / rank
            break

    return {
        "hit_rate": hit_rate,
        "mrr": round(mrr, 4),
        "avg_similarity": round(avg_sim, 4),
        "retrieved_sources": [_extract_source_name(d) for d, _ in docs_with_scores],
    }


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
    from config import settings
    from domain.value_objects.llm_provider import LLMProvider

    from infrastructure.benchmark.judge import (
        CONTEXT_PRECISION_PROMPT,
        CONTEXT_RECALL_PROMPT,
        get_judge_client,
        get_judge_model,
        judge_with_structured_output,
    )

    if not context_override and not docs_with_scores:
        return {
            "context_precision": None,
            "context_precision_reason": "No documents retrieved",
            "context_recall": None,
            "context_recall_reason": "No documents retrieved",
        }

    model = judge_model or get_judge_model()

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
            client = get_judge_client(model if settings.llm_provider == LLMProvider.OLLAMA else "")
            score = judge_with_structured_output(
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


# Re-exported from domain for backward compatibility
from domain.utils import percentile as _percentile  # noqa: E402, F401


# LLM pricing per 1M tokens (USD). Ollama = 0 (self-hosted).
LLM_PRICING: dict[str, dict[str, float]] = {
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.05, "output": 0.05},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.52, "output": 0.75},
    "meta-llama/llama-3.1-405b-instruct": {"input": 2.0, "output": 3.0},
}


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD from token counts and model pricing."""
    if model in ("ollama", "") or "ollama" in model.lower():
        return 0.0
    pricing = LLM_PRICING.get(model, {"input": 0.5, "output": 0.5})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def compute_summary_metrics(results: list[dict]) -> dict:
    """Build persisted metrics using the shared benchmark aggregation."""
    return summarize_benchmark_results(results)


def _safe_avg(vals) -> float:
    vals = list(vals)
    return round(sum(vals) / len(vals), 4) if vals else 0
