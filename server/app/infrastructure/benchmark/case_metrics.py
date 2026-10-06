"""Metrics derived from generated benchmark answers and provider usage metadata."""

import logging

from infrastructure.benchmark.source_hints import matches_source_hints, parse_source_hints

logger = logging.getLogger("default")


def compute_retriever_metrics_from_sources(
    sources: list[dict],
    source_hint: str | None,
) -> dict:
    """Compute retriever metrics from RagResult.sources format.

    Sources from RagService are list[dict] with keys: source, max_score, pages, etc.
    """
    scores = [s.get("max_score", 0.0) for s in sources]
    avg_sim = sum(scores) / len(scores) if scores else 0.0

    hints = parse_source_hints(source_hint)
    if not hints:
        return {
            "hit_rate": None,
            "mrr": None,
            "avg_similarity": round(avg_sim, 4),
            "retrieved_sources": [s.get("source", "?") for s in sources],
        }

    hit_rate = 0
    mrr = 0.0
    for rank, src in enumerate(sources, 1):
        filename = src.get("source", "")
        if matches_source_hints(filename, hints):
            hit_rate = 1
            if mrr == 0.0:
                mrr = 1.0 / rank
            break

    return {
        "hit_rate": hit_rate,
        "mrr": round(mrr, 4),
        "avg_similarity": round(avg_sim, 4),
        "retrieved_sources": [s.get("source", "?") for s in sources],
    }


def _extract_usage_from_response(response) -> tuple[int | None, int | None]:
    """Extract input/output token counts from a LangChain LLM response."""
    try:
        metadata = getattr(response, "response_metadata", {}) or {}
        token_usage = metadata.get("token_usage", {})
        if token_usage:
            return token_usage.get("prompt_tokens"), token_usage.get("completion_tokens")
        usage = metadata.get("usage", {})
        if usage:
            return usage.get("prompt_tokens"), usage.get("completion_tokens")
    except Exception:
        logger.debug("Failed to extract token usage from LLM response", exc_info=True)
    return None, None
