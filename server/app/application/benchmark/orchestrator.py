"""Benchmark service -- shared async orchestration logic used by the API endpoint.

Delegates the actual benchmark execution to ``BenchmarkRunnerPort``
and handles result persistence, history tracking, and regression comparison.
"""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING, Any, Unpack

from application.benchmark.summary import summarize_benchmark_results
from domain.benchmark.diagnosis import with_diagnosis
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
            top_k=top_k,
            judge_model=judge_model,
            max_concurrent=options.get("max_concurrent"),
            space_id=options.get("space_id"),
            seed=options.get("seed"),
            n_runs=options.get("n_runs", 1),
            rag_service=self._rag_service,
            resume=options.get("resume", False),
            checkpoints=options.get("checkpoints"),
            checkpoint_prefix=options.get("checkpoint_prefix"),
            depth=options.get("depth"),
        )
        return {**compute_summary_from_results(results), "status": "done"}


def compute_summary_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build summary dict from benchmark results list."""
    results = with_diagnosis(results)
    breadths = [r.get("breadth") for r in results if r.get("breadth") is not None]
    answer_chars = sorted(len(r.get("answer") or "") for r in results)
    finish_reasons = [r.get("finish_reason") for r in results if r.get("finish_reason") is not None]

    return {
        **summarize_benchmark_results(results),
        "breadth_distribution": {
            "narrow": sum(1 for b in breadths if b == "narrow"),
            "broad": sum(1 for b in breadths if b == "broad"),
        }
        if breadths
        else None,
        # Answer length per style is the main signal for the "Кратко / Авто / Подробно" selector.
        "answer_length": {
            "mean_chars": round(sum(answer_chars) / len(answer_chars), 1),
            "median_chars": statistics.median(answer_chars),
            "max_chars": answer_chars[-1],
        }
        if answer_chars
        else None,
        "truncated_rate": round(
            sum(1 for reason in finish_reasons if reason == "length") / len(finish_reasons), 4
        )
        if finish_reasons
        else None,
        "results": [
            {
                **r,
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
                "generation_sec": r.get("generation_sec"),
                "ttft_sec": r.get("ttft_sec"),
                "llm_replayed": r.get("llm_replayed", False),
                "breadth": r.get("breadth"),
                "answer_style": r.get("answer_style"),
                "finish_reason": r.get("finish_reason"),
                "answer_chars": len(r.get("answer") or ""),
            }
            for r in results
        ],
    }
