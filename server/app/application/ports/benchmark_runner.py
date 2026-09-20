"""Benchmark runner port — application-layer abstraction for running RAG benchmarks.

Decouples the application service from the concrete benchmark runner in
infrastructure, following the same pattern as other ports in this package.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from application.ports.chat_rag_port import ChatRAGPort


@runtime_checkable
class BenchmarkRunnerPort(Protocol):
    """Runs a benchmark evaluation against the RAG pipeline."""

    async def run(
        self,
        questions_path: str,
        out_dir: str,
        top_k: int,
        judge_model: str,
        max_concurrent: int = 4,
        seed: int | None = None,
        n_runs: int = 1,
        rag_service: ChatRAGPort | None = None,
    ) -> None: ...
