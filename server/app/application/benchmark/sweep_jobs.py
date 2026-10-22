"""Durable sweep job state, result persistence and completion policies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from application.benchmark.identity import fingerprint
from domain.benchmark.evaluation import EVIDENCE_METRICS
from domain.benchmark.run import BenchmarkRun
from domain.benchmark.sweep import BenchmarkSweepStatus


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from application.benchmark.export_service import BenchmarkExportService
    from application.uow.factory import UnitOfWorkFactory
    from domain.benchmark.sweep import BenchmarkSweep

logger = logging.getLogger(__name__)


class IncompleteEvaluationError(ValueError):
    """All work finished, but some judge scores remain unavailable."""


def validate_sweep_coverage(results: list[dict[str, Any]]) -> None:
    evaluated = [r["full_metrics"] for r in results if r.get("llm_evaluated")]
    error_count = sum(r.get("judge_error_count", 0) for r in evaluated)
    incomplete = sum(r.get("evaluation_complete") is False for r in results)

    if error_count or incomplete:
        total = sum(r.get("total_questions", 0) for r in evaluated)
        complete = sum(r.get("judge_evaluated_count", 0) for r in evaluated)
        # All configurations have run. Retain checkpoints and expose Resume
        # using the existing failed-sweep lifecycle, even with a valid winner.
        message = (
            f"Оценено {complete} из {total}, ошибок {error_count}. "
            f"Неполных конфигураций: {incomplete}. Resume выполнит недостающие оценки."
        )
        raise IncompleteEvaluationError(message)


class SweepJobService:
    """Persist sweep progress before notifying observers."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        publish: Callable[[int, dict[str, Any]], Awaitable[None]],
        exports: BenchmarkExportService,
    ) -> None:
        self.uow_factory = uow_factory
        self.publish = publish
        self.exports = exports

    async def handle_sweep_cancelled(self, sweep_id: int) -> None:
        logger.info("Sweep %d cancelled — stopping early", sweep_id)
        async with self.uow_factory.create(master=True) as uow:
            await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.CANCELLED.value)
        await self.publish(sweep_id, {"done": True, "cancelled": True})

    async def save_sweep_results(
        self, sweep: BenchmarkSweep, sweep_id: int, results: list[dict[str, Any]]
    ) -> int | None:
        """Persist every config, including partial results; return the eligible winner."""
        best_run_id = None

        for result in results:
            run_id = await self.save_sweep_run(sweep, sweep_id, result)

            if best_run_id is None and result.get("composite_score") is not None:
                best_run_id = run_id

        async with self.uow_factory.create(master=True) as uow:
            if best_run_id is not None:
                await uow.benchmark_sweeps.set_best_run(sweep_id, best_run_id)
            # Completion is set only after checking coverage and cancellation.
        return best_run_id

    async def save_sweep_run(self, sweep: BenchmarkSweep, sweep_id: int, best: dict[str, Any]) -> int | None:
        config = best.get("config", {})
        full = best.get("full_metrics", {})
        metrics = {
            "hit_rate": full.get("hit_rate") if full else best.get("avg_hit_rate"),
            "mrr": full.get("avg_mrr") if full else best.get("avg_mrr"),
            "composite": best.get("composite_score"),
            "faithfulness": best.get("full_metrics", {}).get("avg_faithfulness"),
            "relevancy": full.get("avg_relevancy"),
            "correctness": full.get("avg_correctness"),
            "evaluation_mode": sweep.evaluation_mode,
            "evaluated_config_count": best.get("evaluated_config_count", 0),
            "scored_config_count": best.get("scored_config_count", 0),
            "search_config_count": best.get("search_config_count", 0),
            **{
                key: full.get(key)
                for key in (
                    "total_questions",
                    "judge_evaluated_count",
                    "judge_error_count",
                    "total_input_tokens",
                    "total_output_tokens",
                    "total_judge_input_tokens",
                    "total_judge_output_tokens",
                    "total_judge_calls",
                    "usage_scope",
                )
            },
        }
        # Keep evidence averages and coverage counts so a composite score can be
        # explained without reopening the original report files.
        diagnostics = full or best
        metrics.update(
            {
                key: value
                for key, value in diagnostics.items()
                if key in {f"avg_{metric}" for metric in EVIDENCE_METRICS}
                or key.startswith(
                    ("avg_fragment_", "avg_context_", "avg_retrieval_", "avg_source_", "retrieval_")
                )
                or key.endswith(("_evaluated_count", "_expected_count"))
            }
        )
        async with self.uow_factory.create(master=True) as uow:
            runtime = await uow.benchmark_sweeps.load_checkpoint(sweep_id, "runtime.json")
            dataset_snapshot = await uow.benchmark_sweeps.load_checkpoint(sweep_id, "dataset.json")

            if runtime is not None:
                runtime = cast("dict[str, Any]", runtime)
                metrics.update(
                    collection_name=runtime.get("collection_name"),
                    qdrant_url=runtime.get("qdrant_url"),
                    corpus_generation=runtime.get("corpus_generation"),
                    corpus_namespace=runtime.get("corpus_namespace"),
                    embedding_identity=runtime.get("embedding_identity"),
                )

            if dataset_snapshot is not None:
                metrics["dataset_hash"] = fingerprint(dataset_snapshot)

            run_entity = BenchmarkRun(
                sweep_id=sweep_id,
                config_json=config,
                summary_metrics=metrics,
                dataset=sweep.dataset,
                llm_evaluated=best.get("llm_evaluated", False),
                per_question_results=best.get("full_metrics", {}).get("results"),
            )
            run_entity = await uow.benchmark_runs.save_for_sweep(run_entity)
        return run_entity.id

    async def complete_sweep(
        self, sweep_id: int, results: list[dict[str, Any]], best_run_id: int | None
    ) -> None:
        async with self.uow_factory.create(master=True) as uow:
            current = await uow.benchmark_sweeps.get_by_id(sweep_id, for_update=True)

            cancelled = current is not None and current.status == BenchmarkSweepStatus.CANCELLED.value

            if not cancelled:
                await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.DONE.value)

        if cancelled:
            await self.publish(sweep_id, {"done": True, "cancelled": True})
            return

        try:
            async with self.uow_factory.create(master=True) as uow:
                await uow.benchmark_sweeps.delete_checkpoints(sweep_id)
        except Exception:
            logger.exception("Sweep checkpoint cleanup deferred to cron: sweep=%s", sweep_id)
        await self.publish(
            sweep_id,
            {"done": True, "best_run_id": best_run_id, "total_results": len(results)},
        )
        logger.info("Sweep %d completed: %d results, best_run_id=%s", sweep_id, len(results), best_run_id)

    async def sweep_job_cancelled(self, sweep_id: int, job_id: int) -> bool:
        async with self.uow_factory.create(master=True) as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)
            return sweep is not None and (
                sweep.status == BenchmarkSweepStatus.CANCELLED.value
                or (sweep.job_id is not None and sweep.job_id != job_id)
            )

    async def record_sweep_progress(
        self, sweep: BenchmarkSweep, sweep_id: int, ev: int, tot: int, res: dict[str, Any] | None
    ) -> None:
        async with self.uow_factory.create(master=True) as uow:
            await uow.benchmark_sweeps.update_progress(sweep_id, ev, tot)

        if res and res.get("phase") == "full_evaluation" and "config" in res:
            await self.save_sweep_run(sweep, sweep_id, res)

        await self.publish(sweep_id, {"evaluated": ev, "total": tot, "latest": res})

    async def start_sweep_job(self, sweep_id: int, job_id: int) -> BenchmarkSweep | None:
        async with self.uow_factory.create(master=True) as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id, for_update=True)

            if sweep is None:
                message = f"Sweep {sweep_id} not found"
                raise ValueError(message)

            if sweep.status != BenchmarkSweepStatus.PENDING.value or (
                sweep.job_id is not None and sweep.job_id != job_id
            ):
                logger.info("Ignoring inactive or superseded sweep job %d", job_id)
                return None

            await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.RUNNING.value)
            await uow.benchmark_sweeps.update_progress(sweep_id, 0, 0)
        return sweep

    async def record_sweep_failure(self, sweep_id: int, message: str) -> bool:
        async with self.uow_factory.create(master=True) as uow:
            current = await uow.benchmark_sweeps.get_by_id(sweep_id, for_update=True)
            cancelled = current is not None and current.status == BenchmarkSweepStatus.CANCELLED.value

            if not cancelled:
                await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.FAILED.value)
        event: dict[str, Any] = (
            {"done": True, "cancelled": True} if cancelled else {"done": True, "error": message}
        )
        await self.publish(sweep_id, event)
        return cancelled

    async def save_sweep_reports(self, sweep_id: int, count: int) -> None:
        async with self.uow_factory.create(master=True) as uow:
            run_ids = await uow.benchmark_runs.get_sweep_run_ids(sweep_id, limit=count)

        for run_id in run_ids:
            for report_format in ("csv", "json"):
                await self.exports.save_report(run_id, report_format)
