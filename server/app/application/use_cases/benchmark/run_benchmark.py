"""Use Case: RunBenchmark — run RAG benchmark via API."""

from __future__ import annotations

import logging
from pathlib import Path

from domain.repositories.benchmark_settings_repository import BenchmarkSettingsProtocol

log = logging.getLogger("default")


class RunBenchmark:
    def __init__(self, benchmark_service, settings: BenchmarkSettingsProtocol) -> None:
        self._benchmark_service = benchmark_service
        self._settings = settings

    def execute(
        self,
        questions_path: str | None = None,
        out_dir: str | None = None,
        top_k: int | None = None,
        judge_model: str | None = None,
    ) -> dict:
        q_path = questions_path or str(Path(self._settings.data_dir) / "test_questions.json")
        o_dir = out_dir or str(Path(self._settings.data_dir) / "benchmark_results")
        k = top_k or self._settings.retriever_top_k
        judge = judge_model or self._settings.llm_model

        return self._benchmark_service.run(
            questions_path=q_path,
            out_dir=o_dir,
            top_k=k,
            judge_model=judge,
        )
