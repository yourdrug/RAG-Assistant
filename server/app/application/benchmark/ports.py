"""Benchmark execution, checkpoint, and history contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Unpack, runtime_checkable


if TYPE_CHECKING:
    from application.benchmark.dto import BenchmarkRunnerOptions


class BenchmarkCheckpoints(Protocol):
    async def load(self, key: str) -> object | None: ...

    async def save(self, key: str, value: object) -> None: ...


@runtime_checkable
class BenchmarkHistoryPort(Protocol):
    """Port for benchmark history operations (baseline, comparison)."""

    def compare_runs(self, current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class BenchmarkRunnerPort(Protocol):
    """Runs a benchmark evaluation against the RAG pipeline."""

    async def run(
        self,
        questions: list[dict[str, Any]],
        out_dir: str | None,
        top_k: int,
        judge_model: str,
        **options: Unpack[BenchmarkRunnerOptions],
    ) -> list[dict[str, Any]]: ...
