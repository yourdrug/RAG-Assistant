"""Benchmark metrics — retriever metrics, context precision/recall, summary."""

import logging
from pathlib import Path

from langchain.schema import Document

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

    if source_hint is None:
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
        if source_hint.lower() in filename.lower():
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
) -> dict:
    """Compute context_precision and context_recall via LLM judge.

    Args:
        question: The user question.
        answer: The generated answer.
        docs_with_scores: Retrieved documents with similarity scores.
        judge_llm: Optional pre-configured judge LLM.
        context_override: If provided, use this as the context string directly
            instead of extracting from docs_with_scores.

    """
    from config import settings
    from domain.value_objects.llm_provider import LLMProvider

    from infrastructure.benchmark.judge import (
        CONTEXT_PRECISION_PROMPT,
        CONTEXT_RECALL_PROMPT,
        _get_judge_client,
        _get_judge_model,
        _judge_with_structured_output,
    )

    if not context_override and not docs_with_scores:
        return {
            "context_precision": None,
            "context_precision_reason": "No documents retrieved",
            "context_recall": None,
            "context_recall_reason": "No documents retrieved",
        }

    model = _get_judge_model()
    client = _get_judge_client(model if settings.llm_provider == LLMProvider.OLLAMA else "")

    if context_override:
        context = context_override
    else:
        if docs_with_scores is None:
            raise ValueError("docs_with_scores must be provided when context_override is not set")
        context = "\n\n---\n\n".join(
            d.page_content if hasattr(d, "page_content") else str(d) for d, _ in docs_with_scores
        )

    result: dict[str, float | str | None] = {
        "context_precision": None,
        "context_precision_reason": "",
        "context_recall": None,
    }

    try:
        prompt = CONTEXT_PRECISION_PROMPT.format(question=question, answer=answer, context=context)
        cp = _judge_with_structured_output(client, prompt, model)
        result["context_precision"] = max(0.0, min(10.0, cp.score))
        result["context_precision_reason"] = cp.reason
    except Exception as exc:
        logger.warning("Context precision judge failed: %s", exc)
        result["context_precision_reason"] = f"[Error: {exc}]"

    try:
        prompt = CONTEXT_RECALL_PROMPT.format(question=question, answer=answer, context=context)
        cr = _judge_with_structured_output(client, prompt, model)
        result["context_recall"] = max(0.0, min(10.0, cr.score))
        result["context_recall_reason"] = cr.reason
    except Exception as exc:
        logger.warning("Context recall judge failed: %s", exc)
        result["context_recall_reason"] = f"[Error: {exc}]"

    return result


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute percentile from pre-sorted data using linear interpolation."""
    if not sorted_data:
        return 0.0
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


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
    """Extract aggregated metrics from benchmark results.

    Includes latency percentiles, token cost, and coverage diagnostics.
    """
    faiths = [r["generator_metrics"]["faithfulness"] for r in results]
    rels = [r["generator_metrics"]["relevancy"] for r in results]
    corrs = [
        r["generator_metrics"]["correctness"]
        for r in results
        if r["generator_metrics"]["correctness"] is not None
    ]
    hit_rates = [
        r["retriever_metrics"]["hit_rate"] for r in results if r["retriever_metrics"]["hit_rate"] is not None
    ]
    mrrs = [r["retriever_metrics"]["mrr"] for r in results if r["retriever_metrics"]["mrr"] is not None]
    sims = [r["retriever_metrics"]["avg_similarity"] for r in results]

    cp_scores = [
        r["context_metrics"]["context_precision"]
        for r in results
        if r.get("context_metrics", {}).get("context_precision") is not None
    ]
    cr_scores = [
        r["context_metrics"]["context_recall"]
        for r in results
        if r.get("context_metrics", {}).get("context_recall") is not None
    ]

    latencies = sorted(r["latency_sec"] for r in results)
    total_with_expected = sum(1 for r in results if r.get("expected_answer") is not None)
    total_input_tokens = sum(r.get("input_tokens") or 0 for r in results)
    total_output_tokens = sum(r.get("output_tokens") or 0 for r in results)
    total_cost = sum(r.get("cost_usd") or 0.0 for r in results)

    return {
        "total_questions": len(results),
        "pct_with_expected_answer": round(total_with_expected / len(results) * 100, 1) if results else 0,
        "total_time_sec": round(sum(latencies), 1),
        "hit_rate": round(sum(hit_rates) / len(hit_rates), 3) if hit_rates else None,
        "avg_mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
        "avg_faithfulness": round(sum(faiths) / len(faiths), 1) if faiths else None,
        "avg_relevancy": round(sum(rels) / len(rels), 1) if rels else None,
        "avg_correctness": round(sum(corrs) / len(corrs), 1) if corrs else None,
        "avg_similarity": round(sum(sims) / len(sims), 3) if sims else 0,
        "avg_context_precision": round(sum(cp_scores) / len(cp_scores), 1) if cp_scores else None,
        "avg_context_recall": round(sum(cr_scores) / len(cr_scores), 1) if cr_scores else None,
        "latency_p50": round(_percentile(latencies, 50), 2),
        "latency_p95": round(_percentile(latencies, 95), 2),
        "latency_p99": round(_percentile(latencies, 99), 2),
        "latency_min": round(latencies[0], 2) if latencies else 0,
        "latency_max": round(latencies[-1], 2) if latencies else 0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(total_cost, 6),
    }


def _safe_avg(vals) -> float:
    vals = list(vals)
    return round(sum(vals) / len(vals), 4) if vals else 0
