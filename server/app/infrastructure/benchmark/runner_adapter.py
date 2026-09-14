"""Benchmark runner adapter — implements ``BenchmarkRunnerPort`` via infrastructure runner."""

from __future__ import annotations

from typing import Any

from application.ports.benchmark_runner import BenchmarkRunnerPort


class AsyncBenchmarkRunner:
    """Wraps ``infrastructure.benchmark.runner.run_benchmark_async`` behind the port."""

    async def run(
        self,
        questions_path: str,
        out_dir: str,
        top_k: int,
        judge_model: str,
        max_concurrent: int = 4,
        seed: int | None = None,
        n_runs: int = 1,
        rag_service: Any | None = None,
    ) -> None:
        from infrastructure.benchmark.runner import run_benchmark_async

        await run_benchmark_async(
            questions_path=questions_path,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
            max_concurrent=max_concurrent,
            seed=seed,
            n_runs=n_runs,
            rag_service=rag_service,
        )
