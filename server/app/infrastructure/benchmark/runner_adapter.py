"""Benchmark runner adapter — implements ``BenchmarkRunnerPort`` via infrastructure runner."""

from __future__ import annotations

from typing import TYPE_CHECKING
from application.ports.benchmark_checkpoints import BenchmarkCheckpoints

if TYPE_CHECKING:
    from application.ports.chat_rag_port import ChatRAGPort


class AsyncBenchmarkRunner:
    """Wraps ``infrastructure.benchmark.runner.run_benchmark_async`` behind the port."""

    async def run(
        self,
        questions: list[dict],
        out_dir: str,
        top_k: int,
        judge_model: str,
        max_concurrent: int | None = None,
        seed: int | None = None,
        n_runs: int = 1,
        rag_service: "ChatRAGPort | None" = None,
        resume: bool = False,
        checkpoints: BenchmarkCheckpoints | None = None,
        checkpoint_prefix: str | None = None,
        export_files: bool = False,
    ) -> list[dict]:
        from infrastructure.benchmark.runner import run_benchmark_async

        return await run_benchmark_async(
            questions=questions,
            out_dir=out_dir,
            top_k=top_k,
            judge_model=judge_model,
            max_concurrent=max_concurrent,
            seed=seed,
            n_runs=n_runs,
            rag_service=rag_service,
            resume=resume,
            checkpoints=checkpoints,
            checkpoint_prefix=checkpoint_prefix,
            export_files=export_files,
        )
