"""Benchmark service -- shared async orchestration logic used by the API endpoint.

Delegates the actual benchmark execution to ``infrastructure.benchmark.runner``
and handles result persistence, history tracking, and regression comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import settings

from infrastructure.benchmark.metrics import _percentile
from infrastructure.benchmark.runner import run_benchmark_async

log = logging.getLogger("default")


class BenchmarkService:
    def __init__(self, rag_service=None):
        self._rag_service = rag_service

    async def run(
        self,
        questions_path: str,
        out_dir: str,
        top_k: int,
        judge_model: str,
        seed: int | None = None,
        n_runs: int = 1,
        max_concurrent: int = 4,
    ) -> dict:
        """Run benchmark via shared async implementation, return summary dict."""
        log.info("RAG Benchmark")
        log.info("  questions : %s", questions_path)
        log.info("  top_k     : %d", top_k)
        log.info("  rag model : %s", settings.llm_model)
        log.info("  judge     : %s", judge_model)

        await run_benchmark_async(
            questions_path=questions_path,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
            max_concurrent=max_concurrent,
            seed=seed,
            n_runs=n_runs,
            rag_service=self._rag_service,
        )

        # Read back the latest results JSON to return structured summary
        result_files = sorted(Path(out_dir).glob("benchmark_*.json"))
        if not result_files:
            return {"status": "done", "total_questions": 0}

        latest = json.loads(result_files[-1].read_text(encoding="utf-8"))
        summary = compute_summary_from_results(latest)
        summary["status"] = "done"
        summary["json_path"] = str(result_files[-1])
        return summary


def compute_summary_from_results(results: list[dict]) -> dict:
    """Build summary dict from benchmark results list."""
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
        r.get("context_metrics", {}).get("context_precision")
        for r in results
        if r.get("context_metrics", {}).get("context_precision") is not None
    ]
    cr_scores = [
        r.get("context_metrics", {}).get("context_recall")
        for r in results
        if r.get("context_metrics", {}).get("context_recall") is not None
    ]

    latencies = sorted(r["latency_sec"] for r in results)
    total_input_tokens = sum(r.get("input_tokens") or 0 for r in results)
    total_output_tokens = sum(r.get("output_tokens") or 0 for r in results)
    total_cost = sum(r.get("cost_usd") or 0.0 for r in results)

    breadths = [r.get("breadth") for r in results if r.get("breadth") is not None]

    return {
        "total_questions": len(results),
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
        "breadth_distribution": {
            "narrow": sum(1 for b in breadths if b == "narrow"),
            "broad": sum(1 for b in breadths if b == "broad"),
        }
        if breadths
        else None,
        "results": [
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "expected_answer": r.get("expected_answer"),
                "faithfulness": r["generator_metrics"]["faithfulness"],
                "relevancy": r["generator_metrics"]["relevancy"],
                "correctness": r["generator_metrics"]["correctness"],
                "hit_rate": r["retriever_metrics"]["hit_rate"],
                "mrr": r["retriever_metrics"]["mrr"],
                "avg_similarity": r["retriever_metrics"]["avg_similarity"],
                "context_precision": r.get("context_metrics", {}).get("context_precision"),
                "context_recall": r.get("context_metrics", {}).get("context_recall"),
                "latency_sec": r["latency_sec"],
                "ttft_sec": r.get("ttft_sec"),
                "breadth": r.get("breadth"),
            }
            for r in results
        ],
    }
