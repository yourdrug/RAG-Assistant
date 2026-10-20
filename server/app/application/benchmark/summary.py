"""Shared benchmark aggregation for worker responses and persisted runs."""

from typing import Any

from domain.services.benchmark_evaluation import summarize_evidence, summarize_retrieval
from domain.services.benchmark_judge import judge_coverage
from domain.shared.stats import percentile


def summarize_benchmark_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract aggregated metrics from benchmark results.

    estimated_cost_usd covers generation tokens only. total_cost_usd sums observed
    RAG and judge costs and stays unknown when any result has no cost record.
    """
    faiths = [
        r["generator_metrics"]["faithfulness"]
        for r in results
        if r["generator_metrics"].get("faithfulness") is not None
    ]
    rels = [
        r["generator_metrics"]["relevancy"]
        for r in results
        if r["generator_metrics"].get("relevancy") is not None
    ]
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
    known_cost = sum(r["cost_usd"] for r in results if r.get("cost_usd") is not None)
    cost_complete = all(r.get("cost_usd") is not None for r in results)
    total_cost = known_cost if cost_complete else None
    rag_latencies = sorted(r["rag_latency_sec"] for r in results if r.get("rag_latency_sec") is not None)

    return {
        **summarize_evidence(results),
        "total_questions": len(results),
        **judge_coverage(results),
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
        "latency_p50": round(percentile(latencies, 50), 2),
        "latency_p95": round(percentile(latencies, 95), 2),
        "latency_p99": round(percentile(latencies, 99), 2),
        "latency_min": round(latencies[0], 2) if latencies else 0,
        "latency_max": round(latencies[-1], 2) if latencies else 0,
        **summarize_retrieval(results),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_judge_input_tokens": sum(r.get("judge_input_tokens", 0) for r in results),
        "total_judge_output_tokens": sum(r.get("judge_output_tokens", 0) for r in results),
        "total_judge_calls": sum(r.get("judge_calls", 0) for r in results),
        "usage_scope": "observed RAG model calls plus judge responses, including validation retries",
        "rag_time_sec": round(sum(rag_latencies), 2),
        "rag_latency_p50": round(percentile(rag_latencies, 50), 2) if rag_latencies else None,
        "rag_latency_p95": round(percentile(rag_latencies, 95), 2) if rag_latencies else None,
        "total_cost_usd": round(total_cost, 6) if total_cost is not None else None,
        "known_cost_usd": round(known_cost, 6),
        "cost_complete": cost_complete,
        "estimated_cost_usd": round(sum(r.get("estimated_generation_cost_usd") or 0.0 for r in results), 6),
    }
