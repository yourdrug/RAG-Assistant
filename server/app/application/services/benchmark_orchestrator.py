"""Benchmark service -- shared async orchestration logic used by the API endpoint.

Delegates the actual benchmark execution to ``BenchmarkRunnerPort``
and handles result persistence, history tracking, and regression comparison.
"""

from __future__ import annotations

from domain.services.benchmark_evaluation import summarize_evidence, summarize_retrieval
from domain.services.benchmark_judge import judge_coverage, result_judge_errors
from application.ports.benchmark_checkpoints import BenchmarkCheckpoints

import logging
from typing import TYPE_CHECKING

from domain.utils import percentile as _percentile

if TYPE_CHECKING:
    from application.ports.benchmark_runner import BenchmarkRunnerPort
    from application.ports.chat_rag_port import ChatRAGPort

log = logging.getLogger("default")


class BenchmarkService:
    def __init__(self, rag_service: "ChatRAGPort | None" = None, runner: BenchmarkRunnerPort | None = None):
        self._rag_service = rag_service
        self._runner = runner

    def set_rag_service(self, rag_service: "ChatRAGPort") -> None:
        """Wire rag_service for full-pipeline benchmarking."""
        self._rag_service = rag_service

    async def run(
        self,
        questions: list[dict],
        out_dir: str,
        top_k: int,
        judge_model: str,
        seed: int | None = None,
        n_runs: int = 1,
        max_concurrent: int = 4,
        resume: bool = False,
        checkpoints: BenchmarkCheckpoints | None = None,
        checkpoint_prefix: str | None = None,
    ) -> dict:
        """Run benchmark via shared async implementation, return summary dict."""
        log.info("RAG Benchmark")
        log.info("  questions : %d", len(questions))
        log.info("  top_k     : %d", top_k)
        log.info("  judge     : %s", judge_model)

        if self._runner is None:
            raise RuntimeError("A benchmark runner must be injected")
        results = await self._runner.run(
            questions=questions,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
            max_concurrent=max_concurrent,
            seed=seed,
            n_runs=n_runs,
            rag_service=self._rag_service,
            resume=resume,
            checkpoints=checkpoints,
            checkpoint_prefix=checkpoint_prefix,
            export_files=False,
        )
        return {**compute_summary_from_results(results), "status": "done"}


def compute_summary_from_results(results: list[dict]) -> dict:
    """Build summary dict from benchmark results list."""
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
        **summarize_evidence(results),
        "total_questions": len(results),
        **judge_coverage(results),
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
        "total_judge_input_tokens": sum(r.get("judge_input_tokens", 0) for r in results),
        "total_judge_output_tokens": sum(r.get("judge_output_tokens", 0) for r in results),
        "total_judge_calls": sum(r.get("judge_calls", 0) for r in results),
        "usage_scope": "generation plus observed judge calls; excludes RAG auxiliary calls",
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
                "judge_errors": result_judge_errors(r),
                "generator_metrics": r["generator_metrics"],
                "context_metrics": r.get("context_metrics", {}),
                "faithfulness": r["generator_metrics"]["faithfulness"],
                "relevancy": r["generator_metrics"]["relevancy"],
                "correctness": r["generator_metrics"]["correctness"],
                "hit_rate": r["retriever_metrics"]["hit_rate"],
                "mrr": r["retriever_metrics"]["mrr"],
                "source_hit_rate": r["retriever_metrics"].get("source_hit_rate"),
                "source_mrr": r["retriever_metrics"].get("source_mrr"),
                "avg_similarity": r["retriever_metrics"]["avg_similarity"],
                "context_precision": r.get("context_metrics", {}).get("context_precision"),
                "context_recall": r.get("context_metrics", {}).get("context_recall"),
                "evidence_metrics": r.get("evidence_metrics", {}),
                "evidence_diagnostics": r.get("evidence_diagnostics", {}),
                "evidence": r.get("evidence", {}),
                "latency_sec": r["latency_sec"],
                "ttft_sec": r.get("ttft_sec"),
                "breadth": r.get("breadth"),
            }
            for r in results
        ],
        **summarize_retrieval(results),
    }
