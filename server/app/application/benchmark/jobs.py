"""Execute, persist and export a durable benchmark job."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from application.benchmark.dataset import load_benchmark_questions
from application.benchmark.identity import fingerprint
from application.benchmark.replay_ports import replay_disabled
from application.spaces.resolution import default_space_ref
from domain.benchmark.llm_replay import LLMReplayMode
from domain.benchmark.run import BenchmarkRun
from domain.shared.exceptions import BusinessRuleViolationError


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from application.benchmark.export_service import BenchmarkExportService
    from application.benchmark.orchestrator import BenchmarkService
    from application.benchmark.replay_ports import LLMReplayScope
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
        *,
        replay: LLMReplayScope = replay_disabled,
    ) -> None:
        self.uow_factory = uow_factory
        self.benchmark = benchmark
        self.results = results
        self.exports = exports
        self.corpus_scope = corpus_scope
        self.replay = replay

    async def run(  # noqa: PLR0913 - keyword-only fields of the benchmark job payload
        self,
        *,
        dataset: str,
        top_k: int,
        judge_model: str,
        job_id: int,
        space_id: int | None = None,
        expected_dataset_hash: str | None = None,
        llm_replay_mode: LLMReplayMode = LLMReplayMode.OFF,
        depth: str | None = None,
    ) -> None:
        service = self.results
        existing = await service.get_job_result(job_id)

        if existing is None:
            space = await default_space_ref(self.uow_factory, space_id)
            async with self.corpus_scope() as metadata:
                questions = await load_benchmark_questions(self.uow_factory, dataset)
                dataset_hash = fingerprint(questions)

                if expected_dataset_hash is not None and dataset_hash != expected_dataset_hash:
                    # Edited after the run was queued: the result would not match what was checked.
                    message = f"Questions of dataset {dataset} changed after the run was queued"
                    raise BusinessRuleViolationError(message)

                async with self.replay(llm_replay_mode) as replay:
                    summary = await self.benchmark.run(
                        questions=questions,
                        top_k=top_k,
                        judge_model=judge_model,
                        space_id=space.id,
                        depth=depth,
                    )
                replay_metrics = (
                    {"llm_replay": replay.as_dict()} if llm_replay_mode != LLMReplayMode.OFF else {}
                )
                existing = await service.record_result(
                    BenchmarkRun(
                        job_id=job_id,
                        config_json={
                            "top_k": top_k,
                            "judge_model": judge_model,
                            "evaluation_mode": "full",
                            "depth": depth,
                            "dataset_hash": dataset_hash,
                            **metadata,
                            "space_id": space.id,
                            "pack_key": space.pack_key,
                        },
                        summary_metrics={
                            **{
                                key: value
                                for key, value in summary.items()
                                if key not in {"results", "status"}
                            },
                            **replay_metrics,
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
