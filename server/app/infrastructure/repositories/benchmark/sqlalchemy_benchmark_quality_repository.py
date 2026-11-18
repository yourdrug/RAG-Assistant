"""Dataset acceptance policies and human labels in PostgreSQL."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from domain.benchmark.agreement import HumanLabel
from domain.benchmark.policy import (
    AcceptanceThresholds,
    DatasetPolicy,
    JudgeCalibration,
    JudgeIdentity,
    NoiseEstimate,
)
from infrastructure.database.models import (
    benchmark_dataset_policies,
    benchmark_human_labels,
    benchmark_judge_calibrations,
)


if TYPE_CHECKING:
    from sqlalchemy import RowMapping
    from sqlalchemy.ext.asyncio import AsyncSession

    from domain.benchmark.agreement import LabelMetric


class SQLAlchemyBenchmarkQualityRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_policy(self, dataset: str) -> DatasetPolicy | None:
        row = await self._db.execute(
            select(benchmark_dataset_policies).where(benchmark_dataset_policies.c.dataset == dataset)
        )
        found = row.mappings().one_or_none()
        return to_policy(found) if found is not None else None

    async def list_policies(self) -> list[DatasetPolicy]:
        rows = await self._db.execute(
            select(benchmark_dataset_policies).order_by(benchmark_dataset_policies.c.dataset)
        )
        return [to_policy(row) for row in rows.mappings().all()]

    async def save_policy(self, policy: DatasetPolicy) -> DatasetPolicy:
        policy.update_date = datetime.now(UTC)
        values = {
            "thresholds": policy.thresholds.to_dict(),
            "noise": policy.noise.to_dict() if policy.noise is not None else None,
            "calibration": policy.calibration.to_dict() if policy.calibration is not None else None,
            "updated_by": policy.updated_by,
            "update_date": policy.update_date,
        }
        await self._db.execute(
            insert(benchmark_dataset_policies)
            .values(dataset=policy.dataset, **values)
            .on_conflict_do_update(index_elements=["dataset"], set_=values)
        )
        return policy

    async def add_calibration(self, dataset: str, calibration: JudgeCalibration) -> JudgeCalibration:
        calibration_id = await self._db.scalar(
            insert(benchmark_judge_calibrations)
            .values(
                dataset=dataset,
                run_id=calibration.run_id,
                judge_model=calibration.judge.model,
                judge_rubric_version=calibration.judge.rubric_version,
                metrics=calibration.metrics,
                created_by=calibration.created_by,
                measured_at=calibration.measured_at,
            )
            .returning(benchmark_judge_calibrations.c.id)
        )
        return replace(calibration, id=calibration_id)

    async def list_calibrations(self, dataset: str, limit: int) -> list[JudgeCalibration]:
        table = benchmark_judge_calibrations
        rows = await self._db.execute(
            select(table)
            .where(table.c.dataset == dataset)
            .order_by(table.c.measured_at.desc(), table.c.id.desc())
            .limit(limit)
        )
        return [
            JudgeCalibration(
                id=row["id"],
                run_id=row["run_id"],
                judge=JudgeIdentity(model=row["judge_model"], rubric_version=row["judge_rubric_version"]),
                metrics=row["metrics"],
                measured_at=row["measured_at"],
                created_by=row["created_by"],
            )
            for row in rows.mappings().all()
        ]

    async def list_labels(self, run_id: int) -> list[HumanLabel]:
        rows = await self._db.execute(
            select(benchmark_human_labels)
            .where(benchmark_human_labels.c.run_id == run_id)
            .order_by(benchmark_human_labels.c.id)
        )
        return [HumanLabel(**row) for row in rows.mappings().all()]

    async def save_label(self, label: HumanLabel) -> HumanLabel:
        label.update_date = datetime.now(UTC)
        values = {"score": label.score, "note": label.note, "update_date": label.update_date}
        # Concurrent saves of one case metric by one labeler converge on a single row.
        label.id = await self._db.scalar(
            insert(benchmark_human_labels)
            .values(
                run_id=label.run_id,
                question_key=label.question_key,
                metric=label.metric.value,
                labeler_id=label.labeler_id,
                **values,
            )
            .on_conflict_do_update(constraint="uq_benchmark_human_labels_case", set_=values)
            .returning(benchmark_human_labels.c.id)
        )
        return label

    async def delete_label(
        self, run_id: int, question_key: str, metric: LabelMetric, labeler_id: int
    ) -> bool:
        table = benchmark_human_labels
        result = await self._db.execute(
            delete(table)
            .where(
                table.c.run_id == run_id,
                table.c.question_key == question_key,
                table.c.metric == metric.value,
                table.c.labeler_id == labeler_id,
            )
            .returning(table.c.id)
        )
        return result.first() is not None


def to_policy(row: RowMapping) -> DatasetPolicy:
    return DatasetPolicy(
        dataset=row["dataset"],
        thresholds=AcceptanceThresholds.from_dict(row["thresholds"]),
        noise=NoiseEstimate.from_dict(row["noise"]),
        calibration=JudgeCalibration.from_dict(row["calibration"]),
        updated_by=row["updated_by"],
        update_date=row["update_date"],
    )
