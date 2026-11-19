"""Dataset acceptance policy: thresholds, composition, noise, human labels and judge calibration.

Everything here works on stored questions and runs; nothing calls a model.
"""

from __future__ import annotations

from dataclasses import replace
import json
from typing import TYPE_CHECKING, Any

from application.benchmark.regression import normalize_metrics
from domain.benchmark.agreement import HumanLabel, LabelMetric, judge_agreement
from domain.benchmark.composition import dataset_composition
from domain.benchmark.diagnosis import result_key
from domain.benchmark.policy import (
    MIN_NOISE_RUNS,
    AcceptanceThresholds,
    DatasetPolicy,
    JudgeCalibration,
    NoiseEstimate,
    metric_noise,
    run_judge,
)
from domain.benchmark.reachability import expected_fragments, fragment_reachability, fragment_sources
from domain.knowledge.space import SpaceRef
from domain.shared.exceptions import EntityNotFoundError, ValidationError


if TYPE_CHECKING:
    from application.benchmark.ports import BenchmarkChunkDocumentsPort
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.benchmark.composition import CompositionReport
    from domain.benchmark.reachability import ReachabilityReport
    from domain.benchmark.run import BenchmarkRun


MAX_NOISE_RUNS = 10
MAX_CALIBRATIONS = 100


def dataset_pack(dataset: str) -> str | None:
    """`<pack>-<purpose>` naming: the pack is the part before the first hyphen."""
    pack, separator, _ = dataset.partition("-")
    return pack if separator else None


class BenchmarkQualityService:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, chunk_documents: BenchmarkChunkDocumentsPort | None = None
    ) -> None:
        self._uow_factory = uow_factory
        self._chunk_documents = chunk_documents

    async def get_policy(self, dataset: str) -> DatasetPolicy:
        async with self._uow_factory.create() as uow:
            return await uow.benchmark_quality.get_policy(dataset) or DatasetPolicy(dataset=dataset)

    async def list_policies(self) -> list[DatasetPolicy]:
        async with self._uow_factory.create() as uow:
            return await uow.benchmark_quality.list_policies()

    async def update_thresholds(
        self, dataset: str, thresholds: dict[str, Any], user_id: int
    ) -> DatasetPolicy:
        parsed = AcceptanceThresholds.from_dict(thresholds)
        async with self._uow_factory.create(master=True) as uow:
            policy = await uow.benchmark_quality.get_policy(dataset) or DatasetPolicy(dataset=dataset)
            policy = replace(policy, thresholds=parsed, updated_by=user_id)
            return await uow.benchmark_quality.save_policy(policy)

    async def composition(self, dataset: str) -> CompositionReport:
        async with self._uow_factory.create() as uow:
            questions = await uow.benchmark_questions.list_active(dataset=dataset)

        return dataset_composition(
            [
                {
                    "annotations": question.annotations,
                    "history": question.history,
                    "as_of_date": question.as_of_date,
                }
                for question in questions
            ],
            pack_key=dataset_pack(dataset),
        )

    async def fragment_reachability(
        self, dataset: str, space_id: int | None = None
    ) -> tuple[SpaceRef, ReachabilityReport]:
        """Expected fragments of active questions found whole inside one chunk of the space."""
        if self._chunk_documents is None:
            message = "Fragment reachability needs a chunk document source"
            raise RuntimeError(message)

        async with self._uow_factory.create() as uow:
            questions = await uow.benchmark_questions.list_active(dataset=dataset)
            space = await dataset_space(uow, dataset, space_id)

        sources = {
            source
            for question in questions
            for fragment in expected_fragments(question.annotations)
            for source in fragment_sources(fragment)
        }
        rows: list[dict[str, Any]] = [
            {"id": question.id, "question": question.question, "annotations": question.annotations}
            for question in questions
        ]
        documents = await self._chunk_documents.space_documents(space.id, sources) if sources else []
        return space, fragment_reachability(rows, documents)

    async def measure_noise(self, dataset: str, run_ids: list[int], user_id: int | None) -> DatasetPolicy:
        """Standard deviation of each compared metric over repeated runs of one configuration and judge."""
        unique = list(dict.fromkeys(run_ids))

        if not MIN_NOISE_RUNS <= len(unique) <= MAX_NOISE_RUNS:
            message = f"Noise needs {MIN_NOISE_RUNS}-{MAX_NOISE_RUNS} distinct runs"
            raise ValidationError(message)

        async with self._uow_factory.create(master=True) as uow:
            runs = await uow.benchmark_runs.get_by_ids(unique)
            check_noise_runs(dataset, unique, runs)
            judge = run_judge(runs[0].config_json, runs[0].per_question_results)
            noise = NoiseEstimate(
                sigma=metric_noise([normalize_metrics(run.summary_metrics) for run in runs]),
                judge=judge,
                run_ids=sorted(unique),
            )
            policy = await uow.benchmark_quality.get_policy(dataset) or DatasetPolicy(dataset=dataset)
            return await uow.benchmark_quality.save_policy(replace(policy, noise=noise, updated_by=user_id))

    async def list_labels(self, run_id: int) -> list[HumanLabel]:
        async with self._uow_factory.create() as uow:
            await self.require_run(uow, run_id)
            return await uow.benchmark_quality.list_labels(run_id)

    async def save_label(
        self, *, run_id: int, question_key: str, metric: str, score: float, note: str | None, user_id: int
    ) -> HumanLabel:
        label = HumanLabel(
            run_id=run_id,
            question_key=question_key,
            metric=LabelMetric(metric),
            score=score,
            labeler_id=user_id,
            note=note,
        )
        async with self._uow_factory.create(master=True) as uow:
            run = await self.require_run(uow, run_id)

            if question_key not in {result_key(row) for row in run.per_question_results or []}:
                message = "BenchmarkResult"
                raise EntityNotFoundError(message, question_key)

            return await uow.benchmark_quality.save_label(label)

    async def delete_label(self, *, run_id: int, question_key: str, metric: str, user_id: int) -> bool:
        async with self._uow_factory.create(master=True) as uow:
            await self.require_run(uow, run_id)
            return await uow.benchmark_quality.delete_label(
                run_id, question_key, LabelMetric(metric), user_id
            )

    async def agreement(self, run_id: int) -> dict[str, Any]:
        async with self._uow_factory.create() as uow:
            run = await self.require_run(uow, run_id)
            labels = await uow.benchmark_quality.list_labels(run_id)

        return {
            "run_id": run_id,
            "dataset": run.dataset,
            **run_judge(run.config_json, run.per_question_results).to_dict(),
            "metrics": judge_agreement(run.per_question_results or [], labels),
        }

    async def save_calibration(self, run_id: int, user_id: int) -> DatasetPolicy:
        """Store the run's current judge agreement in its dataset policy."""
        async with self._uow_factory.create(master=True) as uow:
            run = await self.require_run(uow, run_id)
            labels = await uow.benchmark_quality.list_labels(run_id)

            if not labels:
                message = "Label at least one answer before saving the calibration"
                raise ValidationError(message)

            calibration = JudgeCalibration(
                run_id=run_id,
                judge=run_judge(run.config_json, run.per_question_results),
                metrics=judge_agreement(run.per_question_results or [], labels),
                created_by=user_id,
            )
            # The history and the policy's latest calibration change in one transaction.
            await uow.benchmark_quality.add_calibration(run.dataset, calibration)
            policy = await uow.benchmark_quality.get_policy(run.dataset) or DatasetPolicy(dataset=run.dataset)
            return await uow.benchmark_quality.save_policy(
                replace(policy, calibration=calibration, updated_by=user_id)
            )

    async def list_calibrations(self, dataset: str, limit: int = MAX_CALIBRATIONS) -> list[JudgeCalibration]:
        async with self._uow_factory.create() as uow:
            return await uow.benchmark_quality.list_calibrations(dataset, limit)

    @staticmethod
    async def require_run(uow: UnitOfWork, run_id: int) -> BenchmarkRun:
        run = await uow.benchmark_runs.get_by_id(run_id)

        if run is None:
            message = "BenchmarkRun"
            raise EntityNotFoundError(message, run_id)

        return run


async def dataset_space(uow: UnitOfWork, dataset: str, space_id: int | None) -> SpaceRef:
    """The explicit space, else the only space of the dataset's pack."""
    if space_id is not None:
        space = await uow.knowledge_spaces.get_by_id(space_id)

        if space is None:
            message = "KnowledgeSpace"
            raise EntityNotFoundError(message, space_id)

        return SpaceRef(id=space_id, pack_key=space.pack_key)

    pack = dataset_pack(dataset)
    spaces = [space for space in await uow.knowledge_spaces.list_all() if space.pack_key == pack]

    if len(spaces) != 1 or spaces[0].id is None:
        message = f"{len(spaces)} spaces use pack {pack}; choose space_id"
        raise ValidationError(message)

    return SpaceRef(id=spaces[0].id, pack_key=spaces[0].pack_key)


def check_noise_runs(dataset: str, run_ids: list[int], runs: list[BenchmarkRun]) -> None:
    found = {run.id for run in runs}
    missing = [run_id for run_id in run_ids if run_id not in found]

    if missing:
        message = "BenchmarkRuns"
        raise EntityNotFoundError(message, str(missing))

    if any(run.dataset != dataset for run in runs):
        message = f"All runs must belong to dataset {dataset}"
        raise ValidationError(message)

    # A replayed LLM response repeats an earlier answer and score, so it would understate sigma.
    replayed = [run.id for run in runs if run_replayed_llm(run)]

    if replayed:
        message = f"Runs {replayed} reused recorded LLM responses; noise needs runs with --llm-replay off"
        raise ValidationError(message)

    judges = {run_judge(run.config_json, run.per_question_results) for run in runs}

    if len(judges) > 1:
        message = "Runs were judged by different models or rubrics"
        raise ValidationError(message)

    # Repeated runs measure noise only when nothing else changed between them.
    identities = {
        tuple(sorted((key, str(value)) for key, value in run.config_json.items() if key != "job_id"))
        for run in runs
    }

    if len(identities) > 1:
        message = "Runs used different configurations; noise needs repeated runs of one configuration"
        raise ValidationError(message)

    if len({run_identity(run) for run in runs}) > 1:
        message = (
            "Runs differ in RAG configuration, code, templates, corpus or questions; "
            "noise needs repeated runs of one configuration"
        )
        raise ValidationError(message)


# Manifest fields that describe a run without identifying what was evaluated.
NON_IDENTITY_MANIFEST_KEYS = frozenset({"limitations"})


def run_replayed_llm(run: BenchmarkRun) -> bool:
    """Whether any LLM call of the run (generation or judge) was served from the replay store."""
    replay = (run.summary_metrics or {}).get("llm_replay") or {}

    if replay.get("hits", 0) > 0:
        return True

    return any(row.get("llm_replayed") for row in run.per_question_results or [])


def run_identity(run: BenchmarkRun) -> str:
    """Canonical reproducibility manifest shared by every result of a run."""
    rows = run.per_question_results or []
    manifests = {
        json.dumps(
            {
                key: value
                for key, value in (row.get("run_manifest") or {}).items()
                if key not in NON_IDENTITY_MANIFEST_KEYS
            },
            sort_keys=True,
            default=str,
        )
        for row in rows
    }

    if not rows or len(manifests) != 1 or manifests == {"{}"}:
        message = f"Run {run.id} has no single reproducibility manifest; its configuration cannot be verified"
        raise ValidationError(message)

    return manifests.pop()
