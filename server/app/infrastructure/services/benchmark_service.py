"""Benchmark Service — shared benchmark logic used by both API and CLI.

Delegates to infrastructure/ml/benchmark.py for the actual execution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import settings

from infrastructure.ml.benchmark import run_benchmark

log = logging.getLogger("default")


class BenchmarkService:
    def run(
        self,
        questions_path: str,
        out_dir: str,
        top_k: int,
        judge_model: str,
        seed: int | None = None,
        n_runs: int = 1,
    ) -> dict:
        """Run benchmark via shared implementation, return summary dict.

        Both API and CLI call this method — behaviour is identical.
        """
        log.info("RAG Benchmark")
        log.info("  questions : %s", questions_path)
        log.info("  top_k     : %d", top_k)
        log.info("  rag model : %s", settings.llm_model)
        log.info("  judge     : %s", judge_model)

        run_benchmark(
            questions_path=questions_path,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
            seed=seed,
            n_runs=n_runs,
        )

        # Read back the latest results JSON to return structured summary
        result_files = sorted(Path(out_dir).glob("benchmark_*.json"))
        if not result_files:
            return {"status": "done", "total_questions": 0}

        latest = json.loads(result_files[-1].read_text(encoding="utf-8"))
        summary = _build_summary(latest)
        summary["status"] = "done"
        summary["json_path"] = str(result_files[-1])
        return summary


def _build_summary(results: list[dict]) -> dict:
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

    return {
        "total_questions": len(results),
        "total_time_sec": round(sum(r["latency_sec"] for r in results), 1),
        "hit_rate": round(sum(hit_rates) / len(hit_rates), 3) if hit_rates else None,
        "avg_mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else None,
        "avg_faithfulness": round(sum(faiths) / len(faiths), 1) if faiths else None,
        "avg_relevancy": round(sum(rels) / len(rels), 1) if rels else None,
        "avg_correctness": round(sum(corrs) / len(corrs), 1) if corrs else None,
        "avg_similarity": round(sum(sims) / len(sims), 3) if sims else 0,
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
                "latency_sec": r["latency_sec"],
            }
            for r in results
        ],
    }
