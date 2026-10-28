"""Execute, persist and export a durable benchmark job."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from application.benchmark.dataset import load_benchmark_questions
from application.benchmark.identity import fingerprint
from domain.benchmark.run import BenchmarkRun


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from application.benchmark.export_service import BenchmarkExportService
    from application.benchmark.orchestrator import BenchmarkService
    from application.benchmark.result_service import BenchmarkResultService
    from application.uow.factory import UnitOfWorkFactory


logger = logging.getLogger(__name__)


class BenchmarkJobService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        benchmark: BenchmarkService,
        results: BenchmarkResultService,
        exports: BenchmarkExportService,
        corpus_scope: Callable[[], AbstractAsyncContextManager[dict[str, Any]]],
    ) -> None:
        self.uow_factory = uow_factory
        self.benchmark = benchmark
        self.results = results
        self.exports = exports
        self.corpus_scope = corpus_scope

    async def run(self, *, dataset: str, top_k: int, judge_model: str, job_id: int) -> None:
        service = self.results
        existing = await service.get_job_result(job_id)

        if existing is None:
            async with self.corpus_scope() as metadata:
                questions = await load_benchmark_questions(self.uow_factory, dataset)
                summary = await self.benchmark.run(
                    questions=questions,
                    top_k=top_k,
                    judge_model=judge_model,
                )
                existing = await service.record_result(
                    BenchmarkRun(
                        job_id=job_id,
                        config_json={
                            "top_k": top_k,
                            "judge_model": judge_model,
                            "evaluation_mode": "full",
                            "dataset_hash": fingerprint(questions),
                            **metadata,
                        },
                        summary_metrics={
                            key: value for key, value in summary.items() if key not in {"results", "status"}
                        },
                        per_question_results=summary["results"],
                        duration_sec=summary["total_time_sec"],
                        dataset=dataset,
                        llm_evaluated=True,
                    )
                )

        if existing.id is not None:
            exports = self.exports

            for report_format in ("json", "csv"):
                try:
                    await exports.save_report(existing.id, report_format)
                except Exception:
                    # Committed evaluation remains successful even if metadata/storage is unavailable.
                    logger.exception(
                        "Benchmark artifact deferred: run=%s format=%s", existing.id, report_format
                    )
