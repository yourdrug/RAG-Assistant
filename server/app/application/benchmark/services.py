"""Application services for Benchmark Lab — questions, sweeps, runs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from application.benchmark.diagnosis import rediagnose
from application.benchmark.dto import (
    ApplyConfigResult,
)
from application.spaces.resolution import resolve_space
from domain.benchmark.llm_replay import LLMReplayMode
from domain.benchmark.question import BenchmarkQuestion, new_case_id, validate_annotations
from domain.benchmark.question_snapshot import in_snapshot_scope, plan_question_sync, question_key
from domain.benchmark.sweep import (
    BenchmarkStrategy,
    BenchmarkSweep,
    BenchmarkSweepStatus,
    SweepEvaluationMode,
    validate_sweep_mode,
)
from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus, JobType
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError, ValidationError


if TYPE_CHECKING:
    import builtins
    from collections.abc import Sequence

    from application.admin import ConfigService
    from application.benchmark.dto import (
        BenchmarkQuestionCreateDTO,
        RunCompareResult,
        SweepCreateDTO,
    )
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.benchmark.question_snapshot import QuestionSyncPlan
    from domain.benchmark.run import BenchmarkRun


log = logging.getLogger("default")
MIN_COMPARE_RUNS = 2
MAX_COMPARE_RUNS = 10
QUESTION_EXPORT_LIMIT = 10_000

# ---------------------------------------------------------------------------
# Benchmark run config → live config parameter mapping
# ---------------------------------------------------------------------------
RUN_CONFIG_KEY_MAP: dict[str, str] = {
    # Benchmark top_k/fetch_k are the chat rerank and candidate limits: chat reads only the broad settings.
    "top_k": "retriever_top_k_broad",
    "fetch_k": "retriever_fetch_k_broad",
    "dense_weight": "dense_weight",
    "sparse_weight": "sparse_weight",
    "rrf_k": "rrf_k",
    "rerank_min_score": "rerank_min_score",
    "rerank_score_gap_ratio": "rerank_score_gap_ratio",
}

HISTORY_CONFIG_KEYS: tuple[str, ...] = ("top_k", "fetch_k", "dense_weight", "sparse_weight", "rrf_k")


class BenchmarkQuestionService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list(
        self,
        *,
        dataset: str | None = None,
        tag: str | None = None,
        search: str | None = None,
        is_active: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[builtins.list[BenchmarkQuestion], int]:
        async with self._uow_factory.create() as uow:
            questions = await uow.benchmark_questions.list_items(
                dataset=dataset, tag=tag, search=search, is_active=is_active, limit=limit, offset=offset
            )
            total = await uow.benchmark_questions.count(
                dataset=dataset, tag=tag, search=search, is_active=is_active
            )
            return questions, total

    async def list_datasets(self) -> builtins.list[str]:
        async with self._uow_factory.create() as uow:
            return await uow.benchmark_questions.get_datasets()

    async def create(self, body: BenchmarkQuestionCreateDTO, created_by: int) -> BenchmarkQuestion:
        entity = BenchmarkQuestion(
            question=body.question,
            expected_answer=body.expected_answer,
            source_hint=body.source_hint,
            annotations=body.annotations,
            case_id=body.case_id or new_case_id(),
            variant_of=body.variant_of,
            question_style=body.question_style,
            history=body.history,
            as_of_date=body.as_of_date,
            tags=body.tags,
            dataset=body.dataset,
            notes=body.notes,
            is_active=body.is_active,
            created_by=created_by,
        )
        async with self._uow_factory.create(master=True) as uow:
            return await uow.benchmark_questions.create(entity)

    async def update(self, question_id: int, fields: dict[str, Any]) -> BenchmarkQuestion:
        if "annotations" in fields:
            fields = {**fields, "annotations": validate_annotations(fields["annotations"])}

        async with self._uow_factory.create(master=True) as uow:
            updated = await uow.benchmark_questions.update(question_id, **fields)

        if updated is None:
            message = "BenchmarkQuestion"
            raise EntityNotFoundError(message, question_id)

        return updated

    async def delete(self, question_id: int) -> bool:
        async with self._uow_factory.create(master=True) as uow:
            deleted = await uow.benchmark_questions.delete(question_id)

        if not deleted:
            message = "BenchmarkQuestion"
            raise EntityNotFoundError(message, question_id)

        return True

    async def bulk_create(self, bodies: builtins.list[BenchmarkQuestionCreateDTO], created_by: int) -> int:
        entities = [
            BenchmarkQuestion(
                question=q.question,
                expected_answer=q.expected_answer,
                source_hint=q.source_hint,
                annotations=q.annotations,
                case_id=q.case_id or new_case_id(),
                variant_of=q.variant_of,
                question_style=q.question_style,
                history=q.history,
                as_of_date=q.as_of_date,
                tags=q.tags,
                dataset=q.dataset,
                is_active=q.is_active,
                created_by=created_by,
            )
            for q in bodies
        ]
        async with self._uow_factory.create(master=True) as uow:
            return await uow.benchmark_questions.bulk_create(entities)

    async def export(self, dataset: str | None = None) -> builtins.list[BenchmarkQuestion]:
        async with self._uow_factory.create() as uow:
            return await uow.benchmark_questions.list_items(dataset=dataset, limit=QUESTION_EXPORT_LIMIT)

    async def snapshot(self, dataset_prefix: str) -> builtins.list[BenchmarkQuestion]:
        """Stored questions of the datasets `<pack>-*`, in a stable order for a committed file."""
        async with self._uow_factory.create() as uow:
            questions = await self.scoped(uow, dataset_prefix)
        return sorted(questions, key=question_key)

    async def plan_sync(self, snapshot: Sequence[BenchmarkQuestion], dataset_prefix: str) -> QuestionSyncPlan:
        async with self._uow_factory.create() as uow:
            return plan_question_sync(snapshot, await self.scoped(uow, dataset_prefix))

    async def apply_sync(
        self, snapshot: Sequence[BenchmarkQuestion], dataset_prefix: str
    ) -> QuestionSyncPlan:
        """Create and update stored questions from the snapshot in one transaction; never delete."""
        outside = sorted({q.dataset for q in snapshot if not in_snapshot_scope(q, dataset_prefix)})

        if outside:
            message = f"Snapshot datasets outside {dataset_prefix}*: {', '.join(outside)}"
            raise ValidationError(message)

        async with self._uow_factory.create(master=True) as uow:
            plan = plan_question_sync(snapshot, await self.scoped(uow, dataset_prefix))

            if plan.create:
                await uow.benchmark_questions.bulk_create([*plan.create])

            for item in plan.update:
                await uow.benchmark_questions.update(item.question_id, **item.changes)

        return plan

    @staticmethod
    async def scoped(uow: UnitOfWork, dataset_prefix: str) -> builtins.list[BenchmarkQuestion]:
        stored = await uow.benchmark_questions.list_items(limit=QUESTION_EXPORT_LIMIT)

        if len(stored) >= QUESTION_EXPORT_LIMIT:
            message = f"More than {QUESTION_EXPORT_LIMIT} benchmark questions; snapshot would be partial"
            raise BusinessRuleViolationError(message)

        return [question for question in stored if in_snapshot_scope(question, dataset_prefix)]


class BenchmarkSweepService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def create(
        self, body: SweepCreateDTO, *, submit_job: bool = False, cost_principal: str | None = None
    ) -> BenchmarkSweep:
        validate_sweep_mode(body.evaluation_mode, body.strategy)
        async with self._uow_factory.create(master=True) as uow:
            if await uow.benchmark_sweeps.has_active():
                message = "Another sweep is pending or running — cancel it or wait for it to finish"
                raise BusinessRuleViolationError(message)

            space = await resolve_space(uow, body.space_id)
            sweep_entity = BenchmarkSweep(
                space_id=space.id,
                strategy=BenchmarkStrategy(body.strategy),
                search_space=body.search_space,
                objective_weights=body.objective_weights,
                dataset=body.dataset,
                top_n_llm=body.top_n_llm,
                judge_model=body.judge_model,
                evaluation_mode=SweepEvaluationMode(body.evaluation_mode),
                llm_replay_mode=LLMReplayMode(body.llm_replay_mode),
                status=BenchmarkSweepStatus.PENDING,
            )
            sweep = await uow.benchmark_sweeps.create(sweep_entity)

            if submit_job:
                await self.persist_submission(uow, sweep, cost_principal)

            return sweep

    async def get(self, sweep_id: int) -> BenchmarkSweep:
        async with self._uow_factory.create() as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)

        if sweep is None:
            message = "BenchmarkSweep"
            raise EntityNotFoundError(message, sweep_id)

        return sweep

    async def progress(self, sweep_id: int) -> dict[str, Any] | None:
        async with self._uow_factory.create() as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)

            if sweep is None:
                message = "BenchmarkSweep"
                raise EntityNotFoundError(message, sweep_id)

            snapshot = await uow.benchmark_sweeps.load_checkpoint(sweep_id, "progress.json")

        if (
            not isinstance(snapshot, dict)
            or snapshot.get("job_id") != sweep.job_id
            or sweep.status == BenchmarkSweepStatus.PENDING
        ):
            return None

        return snapshot

    async def list(self, limit: int = 50, offset: int = 0) -> tuple[builtins.list[BenchmarkSweep], int]:
        async with self._uow_factory.create() as uow:
            sweeps = await uow.benchmark_sweeps.list_items(limit=limit, offset=offset)
            total = await uow.benchmark_sweeps.count()
            return sweeps, total

    async def cancel(self, sweep_id: int) -> None:
        async with self._uow_factory.create(master=True) as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id, for_update=True)

            if sweep is None:
                message = "BenchmarkSweep"
                raise EntityNotFoundError(message, sweep_id)

            if sweep.status not in (BenchmarkSweepStatus.PENDING.value, BenchmarkSweepStatus.RUNNING.value):
                message = f"Cannot cancel sweep in '{sweep.status}' status"
                raise ValidationError(message)

            await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.CANCELLED.value)

    async def resume(
        self, sweep_id: int, *, submit_job: bool = False, cost_principal: str | None = None
    ) -> BenchmarkSweep:
        async with self._uow_factory.create(master=True) as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)

            if sweep is None:
                message = "BenchmarkSweep"
                raise EntityNotFoundError(message, sweep_id)

            if sweep.job_id is not None:
                job = await uow.background_jobs.get_by_id(sweep.job_id)

                if job is not None and job.status in (
                    BackgroundJobStatus.PENDING.value,
                    BackgroundJobStatus.RUNNING.value,
                ):
                    message = "The previous sweep job is still stopping; retry after it finishes"
                    raise BusinessRuleViolationError(message)

            if not await uow.benchmark_sweeps.requeue(sweep_id):
                message = "Only failed or cancelled sweeps can be resumed"
                raise ValidationError(message)

            sweep.status = BenchmarkSweepStatus.PENDING
            sweep.job_id = None

            if submit_job:
                await self.persist_submission(uow, sweep, cost_principal)

            return sweep

    async def persist_submission(
        self, uow: UnitOfWork, sweep: BenchmarkSweep, cost_principal: str | None = None
    ) -> None:
        if sweep.id is None:
            message = "Cannot submit an unsaved sweep"
            raise RuntimeError(message)

        job = await uow.background_jobs.create(
            BackgroundJob(
                job_type=JobType.SWEEP,
                related_id=sweep.id,
                dispatch_payload={"sweep_id": sweep.id, "cost_principal": cost_principal},
            )
        )

        if job.id is None:
            message = "BackgroundJob create returned None id"
            raise RuntimeError(message)

        await uow.benchmark_sweeps.set_job_id(sweep.id, job.id)
        sweep.job_id = job.id

    async def attach_job(self, sweep_id: int, job_id: int) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.benchmark_sweeps.set_job_id(sweep_id, job_id)

    async def update_status(self, sweep_id: int, status: str) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.benchmark_sweeps.update_status(sweep_id, status)


class BenchmarkRunService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list(
        self,
        *,
        sweep_id: int | None = None,
        dataset: str | None = None,
        sort_by: str = "creation_date",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[builtins.list[BenchmarkRun], int]:
        async with self._uow_factory.create() as uow:
            runs = await uow.benchmark_runs.list_items(
                sweep_id=sweep_id,
                dataset=dataset,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )
            total = await uow.benchmark_runs.count(sweep_id=sweep_id, dataset=dataset)
            return runs, total

    async def get(self, run_id: int) -> BenchmarkRun:
        async with self._uow_factory.create() as uow:
            run = await uow.benchmark_runs.get_by_id(run_id)

        if run is None:
            message = "BenchmarkRun"
            raise EntityNotFoundError(message, run_id)

        return run

    async def get_by_ids(self, ids: Sequence[int]) -> builtins.list[BenchmarkRun]:
        async with self._uow_factory.create() as uow:
            return await uow.benchmark_runs.get_review_runs(list(ids))

    async def compare(self, ids: Sequence[int]) -> RunCompareResult:
        if len(ids) < MIN_COMPARE_RUNS:
            message = "Provide at least 2 run IDs"
            raise ValidationError(message)

        if len(ids) > MAX_COMPARE_RUNS:
            message = "Maximum 10 runs for comparison"
            raise ValidationError(message)

        runs = [rediagnose(run) for run in await self.get_by_ids(ids)]

        if len(runs) != len(ids):
            found_ids = {r.id for r in runs}
            missing = [i for i in ids if i not in found_ids]
            message = "BenchmarkRuns"
            raise EntityNotFoundError(message, str(missing))

        diff: dict[str, list[dict[str, Any]]] = {}
        all_keys: set[str] = set()

        for r in runs:
            all_keys.update(r.config_json.keys())

        for key in sorted(all_keys):
            diff[key] = [{"run_id": r.id, "value": r.config_json.get(key)} for r in runs]

        return runs, diff

    async def apply_config(
        self,
        run_id: int,
        changed_by: int,
        config_service: ConfigService,
    ) -> ApplyConfigResult:
        """Apply a run's config_json to the live system via ConfigService."""
        run = await self.get(run_id)
        config = run.config_json
        applied_keys: list[str] = []
        failed_keys: list[dict[str, Any]] = []

        values = {
            param_key: str(config[config_key])
            for config_key, param_key in RUN_CONFIG_KEY_MAP.items()
            if config_key in config
        }

        if values:
            # One validated change: the benchmark checked these limits together (e.g. top_k <= fetch_k),
            # so applying only part of them would put an unverified combination into the chat.
            try:
                await config_service.update_parameters(values, changed_by=changed_by)
                applied_keys.extend(values)
            except Exception as e:
                log.warning("Failed to apply run %d config %s: %s", run_id, sorted(values), e, exc_info=True)
                failed_keys.extend({"key": key, "error": str(e)} for key in values)

        return ApplyConfigResult(applied=len(applied_keys), keys=applied_keys, failed=failed_keys)
