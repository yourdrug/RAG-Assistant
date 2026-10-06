"""Application services for Benchmark Lab — questions, sweeps, runs."""

from __future__ import annotations

import builtins
import logging
from collections.abc import Sequence
from typing import TypeAlias

from application.dto.benchmark_dto import (
    ApplyConfigResult,
    BenchmarkQuestionCreateDTO,
    SweepCreateDTO,
)
from application.ports.unit_of_work_factory import UnitOfWorkFactory
from application.services.config_service import ConfigService
from domain.entities.benchmark_question import BenchmarkQuestion
from domain.value_objects.benchmark_annotations import validate_annotations
from domain.entities.benchmark_run import BenchmarkRun
from domain.entities.benchmark_sweep import BenchmarkSweep
from domain.exceptions import BusinessRuleViolation, EntityNotFound, ValidationError
from domain.value_objects.sweep_status import BenchmarkSweepStatus
from domain.value_objects.sweep_evaluation_mode import validate_sweep_mode

RunCompareResult: TypeAlias = tuple[list[BenchmarkRun], dict[str, list[dict]]]

log = logging.getLogger("default")

# ---------------------------------------------------------------------------
# Benchmark run config → live config parameter mapping
# ---------------------------------------------------------------------------
RUN_CONFIG_KEY_MAP: dict[str, str] = {
    "top_k": "retriever_top_k",
    "fetch_k": "retriever_fetch_k",
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

    async def create(self, body: BenchmarkQuestionCreateDTO, created_by: int) -> BenchmarkQuestion:
        entity = BenchmarkQuestion(
            question=body.question,
            expected_answer=body.expected_answer,
            source_hint=body.source_hint,
            annotations=body.annotations,
            tags=body.tags,
            dataset=body.dataset,
            notes=body.notes,
            is_active=body.is_active,
            created_by=created_by,
        )
        async with self._uow_factory.create(master=True) as uow:
            return await uow.benchmark_questions.create(entity)

    async def update(self, question_id: int, fields: dict) -> BenchmarkQuestion:
        if "annotations" in fields:
            fields = {**fields, "annotations": validate_annotations(fields["annotations"])}
        async with self._uow_factory.create(master=True) as uow:
            updated = await uow.benchmark_questions.update(question_id, **fields)

        if updated is None:
            raise EntityNotFound("BenchmarkQuestion", question_id)

        return updated

    async def delete(self, question_id: int) -> bool:
        async with self._uow_factory.create(master=True) as uow:
            deleted = await uow.benchmark_questions.delete(question_id)
        if not deleted:
            raise EntityNotFound("BenchmarkQuestion", question_id)
        return True

    async def bulk_create(self, bodies: builtins.list[BenchmarkQuestionCreateDTO], created_by: int) -> int:
        entities = [
            BenchmarkQuestion(
                question=q.question,
                expected_answer=q.expected_answer,
                source_hint=q.source_hint,
                annotations=q.annotations,
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
            return await uow.benchmark_questions.list_items(dataset=dataset, limit=10000)


class BenchmarkSweepService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def create(self, body: SweepCreateDTO) -> BenchmarkSweep:
        validate_sweep_mode(body.evaluation_mode, body.strategy)
        async with self._uow_factory.create(master=True) as uow:
            if await uow.benchmark_sweeps.has_active():
                raise BusinessRuleViolation(
                    "Another sweep is pending or running — cancel it or wait for it to finish"
                )
            sweep_entity = BenchmarkSweep(
                strategy=body.strategy,
                search_space=body.search_space,
                objective_weights=body.objective_weights,
                dataset=body.dataset,
                top_n_llm=body.top_n_llm,
                judge_model=body.judge_model,
                evaluation_mode=body.evaluation_mode,
                status=BenchmarkSweepStatus.PENDING.value,
            )
            sweep = await uow.benchmark_sweeps.create(sweep_entity)
        return sweep

    async def get(self, sweep_id: int) -> BenchmarkSweep:
        async with self._uow_factory.create() as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)

        if sweep is None:
            raise EntityNotFound("BenchmarkSweep", sweep_id)
        return sweep

    async def list(self, limit: int = 50, offset: int = 0) -> tuple[builtins.list[BenchmarkSweep], int]:
        async with self._uow_factory.create() as uow:
            sweeps = await uow.benchmark_sweeps.list_items(limit=limit, offset=offset)
            total = await uow.benchmark_sweeps.count()
            return sweeps, total

    async def cancel(self, sweep_id: int) -> None:
        async with self._uow_factory.create(master=True) as uow:
            sweep = await uow.benchmark_sweeps.get_by_id(sweep_id)

            if sweep is None:
                raise EntityNotFound("BenchmarkSweep", sweep_id)

            if sweep.status not in (BenchmarkSweepStatus.PENDING.value, BenchmarkSweepStatus.RUNNING.value):
                raise ValidationError(f"Cannot cancel sweep in '{sweep.status}' status")

            await uow.benchmark_sweeps.update_status(sweep_id, BenchmarkSweepStatus.CANCELLED.value)

    async def update_status(self, sweep_id: int, status: str) -> None:
        async with self._uow_factory.create(master=True) as uow:
            await uow.benchmark_sweeps.update_status(sweep_id, status)


class BenchmarkRunService:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def list(
        self,
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
            raise EntityNotFound("BenchmarkRun", run_id)

        return run

    async def get_by_ids(self, ids: Sequence[int]) -> builtins.list[BenchmarkRun]:
        async with self._uow_factory.create() as uow:
            return await uow.benchmark_runs.get_by_ids(list(ids))

    async def compare(self, ids: Sequence[int]) -> RunCompareResult:
        if len(ids) < 2:
            raise ValidationError("Provide at least 2 run IDs")
        if len(ids) > 10:
            raise ValidationError("Maximum 10 runs for comparison")

        runs = await self.get_by_ids(ids)
        if len(runs) != len(ids):
            found_ids = {r.id for r in runs}
            missing = [i for i in ids if i not in found_ids]
            raise EntityNotFound("BenchmarkRuns", str(missing))

        diff: dict[str, list[dict]] = {}
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
        failed_keys: list[dict] = []

        for config_key, param_key in RUN_CONFIG_KEY_MAP.items():
            if config_key in config:
                try:
                    await config_service.update_parameter(
                        param_key, str(config[config_key]), changed_by=changed_by
                    )
                    applied_keys.append(param_key)
                except Exception as e:
                    log.warning("Failed to apply %s=%s: %s", param_key, config[config_key], e)
                    failed_keys.append({"key": param_key, "error": str(e)})

        return ApplyConfigResult(applied=len(applied_keys), keys=applied_keys, failed=failed_keys)
