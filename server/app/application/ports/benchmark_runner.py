"""Benchmark runner port — application-layer abstraction for running RAG benchmarks.

Decouples the application service from the concrete benchmark runner in
infrastructure, following the same pattern as other ports in this package.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from application.ports.chat_rag_port import ChatRAGPort
from application.ports.benchmark_checkpoints import BenchmarkCheckpoints


@runtime_checkable
class BenchmarkRunnerPort(Protocol):
    """Runs a benchmark evaluation against the RAG pipeline."""

    async def run(
        self,
        questions: list[dict],
        out_dir: str,
        top_k: int,
        judge_model: str,
        max_concurrent: int | None = None,
        seed: int | None = None,
        n_runs: int = 1,
        rag_service: ChatRAGPort | None = None,
        resume: bool = False,
        checkpoints: BenchmarkCheckpoints | None = None,
        checkpoint_prefix: str | None = None,
        export_files: bool = False,
    ) -> list[dict]: ...
