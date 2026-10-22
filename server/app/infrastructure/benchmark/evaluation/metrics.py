"""Benchmark metrics — retriever metrics, context precision/recall, summary."""

import logging
from pathlib import Path

from application.benchmark.summary import summarize_benchmark_results
from langchain.schema import Document

from infrastructure.benchmark.evaluation.source_hints import (
    matches_source_hints,
    parse_source_hints,
)

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




# LLM pricing per 1M tokens (USD). Ollama = 0 (self-hosted).
LLM_PRICING: dict[str, dict[str, float]] = {
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.05, "output": 0.05},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.52, "output": 0.75},
    "meta-llama/llama-3.1-405b-instruct": {"input": 2.0, "output": 3.0},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
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
