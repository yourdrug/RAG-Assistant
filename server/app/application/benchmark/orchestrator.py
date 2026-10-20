"""Benchmark service -- shared async orchestration logic used by the API endpoint.

Delegates the actual benchmark execution to ``BenchmarkRunnerPort``
and handles result persistence, history tracking, and regression comparison.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Unpack

from application.benchmark.summary import summarize_benchmark_results
from domain.benchmark.judge import result_judge_errors


if TYPE_CHECKING:
    from application.benchmark.dto import BenchmarkRunOptions
    from application.benchmark.ports import BenchmarkRunnerPort
    from application.chat import ChatRAGPort

log = logging.getLogger("default")


class BenchmarkService:
    def __init__(
        self, rag_service: ChatRAGPort | None = None, runner: BenchmarkRunnerPort | None = None
    ) -> None:
        self._rag_service = rag_service
        self._runner = runner

    def set_rag_service(self, rag_service: ChatRAGPort) -> None:
        """Wire rag_service for full-pipeline benchmarking."""
        self._rag_service = rag_service

    async def run(
        self,
        questions: list[dict[str, Any]],
        out_dir: str | None,
        top_k: int,
        judge_model: str,
        **options: Unpack[BenchmarkRunOptions],
    ) -> dict[str, Any]:
        """Run benchmark via shared async implementation, return summary dict."""
        log.info("RAG Benchmark")
        log.info("  questions : %d", len(questions))
        log.info("  top_k     : %d", top_k)
        log.info("  judge     : %s", judge_model)

        if self._runner is None:
            message = "A benchmark runner must be injected"
            raise RuntimeError(message)

        results = await self._runner.run(
            questions=questions,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
            max_concurrent=options.get("max_concurrent"),
            seed=options.get("seed"),
            n_runs=options.get("n_runs", 1),
            rag_service=self._rag_service,
            resume=options.get("resume", False),
            checkpoints=options.get("checkpoints"),
            checkpoint_prefix=options.get("checkpoint_prefix"),
            export_files=False,
        )
        return {**compute_summary_from_results(results), "status": "done"}


def compute_summary_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary dict from benchmark results list."""
    breadths = [r.get("breadth") for r in results if r.get("breadth") is not None]

    return {
        **summarize_benchmark_results(results),
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
    }
