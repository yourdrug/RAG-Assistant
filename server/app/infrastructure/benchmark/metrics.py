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
    docs_with_scores: list[tuple[Document, float]],
    judge_llm=None,
) -> dict:
    """Compute context_precision and context_recall via LLM judge."""
    from config import settings
    from domain.value_objects.llm_provider import LLMProvider

    from infrastructure.benchmark.judge import (
        CONTEXT_PRECISION_PROMPT,
        CONTEXT_RECALL_PROMPT,
        _get_judge_client,
        _judge_with_structured_output,
    )

    if not docs_with_scores:
        return {
            "context_precision": None,
            "context_precision_reason": "No documents retrieved",
            "context_recall": None,
            "context_recall_reason": "No documents retrieved",
        }

    client = _get_judge_client(settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else "")
    model = settings.llm_model if settings.llm_provider == LLMProvider.OLLAMA else settings.openrouter_model

    context = "\n\n---\n\n".join(d.page_content for d, _ in docs_with_scores)

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


def compute_summary_metrics(results: list[dict]) -> dict:
    """Extract aggregated metrics from benchmark results."""
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
        if r["context_metrics"].get("context_precision") is not None
    ]
    cr_scores = [
        r["context_metrics"]["context_recall"]
        for r in results
        if r["context_metrics"].get("context_recall") is not None
    ]

    return {
        "total_questions": len(results),
        "total_time_sec": round(sum(r["latency_sec"] for r in results), 1),
        "hit_rate": round(sum(hit_rates) / len(hit_rates), 3) if hit_rates else None,
        "avg_mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
        "avg_faithfulness": round(sum(faiths) / len(faiths), 1) if faiths else None,
        "avg_relevancy": round(sum(rels) / len(rels), 1) if rels else None,
        "avg_correctness": round(sum(corrs) / len(corrs), 1) if corrs else None,
        "avg_similarity": round(sum(sims) / len(sims), 3) if sims else 0,
        "avg_context_precision": round(sum(cp_scores) / len(cp_scores), 1) if cp_scores else None,
        "avg_context_recall": round(sum(cr_scores) / len(cr_scores), 1) if cr_scores else None,
    }


def _safe_avg(vals) -> float:
    vals = list(vals)
    return round(sum(vals) / len(vals), 4) if vals else 0
