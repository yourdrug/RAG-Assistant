"""SQLAlchemy implementation of BenchmarkRunRepository."""

from __future__ import annotations

from domain.benchmark.run import BenchmarkRun
from domain.benchmark.evaluation import EVIDENCE_METRICS, summarize_evidence
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.database.models import BenchmarkRunModel, BenchmarkSweepModel


class SQLAlchemyBenchmarkRunRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_by_id(self, run_id: int) -> BenchmarkRun | None:
        stmt = select(BenchmarkRunModel).where(BenchmarkRunModel.id == run_id)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self.to_entity(orm) if orm else None

    async def create(self, run: BenchmarkRun) -> BenchmarkRun:
        orm = BenchmarkRunModel(
            job_id=run.job_id,
            legacy_hash=run.legacy_hash,
            artifacts=run.artifacts,
            sweep_id=run.sweep_id,
            config_json=run.config_json,
            summary_metrics=run.summary_metrics,
            duration_sec=run.duration_sec,
            llm_evaluated=run.llm_evaluated,
            dataset=run.dataset,
            per_question_results=run.per_question_results,
            filename=run.filename,
        )
        self._db.add(orm)
        await self._db.flush()
        await self._db.refresh(orm)
        return self.to_entity(orm)

    async def get_by_job_id(self, job_id: int) -> BenchmarkRun | None:
        row = await self._db.scalar(select(BenchmarkRunModel).where(BenchmarkRunModel.job_id == job_id))
        return self.to_entity(row) if row else None

    async def get_by_legacy_hash(self, digest: str) -> BenchmarkRun | None:
        row = await self._db.scalar(select(BenchmarkRunModel).where(BenchmarkRunModel.legacy_hash == digest))
        return self.to_entity(row) if row else None

    async def save_once(self, run: BenchmarkRun) -> BenchmarkRun:
        identity = "job_id" if run.job_id is not None else "legacy_hash"
        if getattr(run, identity) is None:
            raise ValueError("A job or legacy import identity is required")
        values = {
            name: getattr(run, name)
            for name in (
                "job_id",
                "legacy_hash",
                "artifacts",
                "config_json",
                "summary_metrics",
                "duration_sec",
                "llm_evaluated",
                "dataset",
                "per_question_results",
                "filename",
                "creation_date",
            )
        }
        await self._db.execute(
            insert(BenchmarkRunModel).values(**values).on_conflict_do_nothing(index_elements=[identity])
        )
        row = await self._db.scalar(
            select(BenchmarkRunModel).where(getattr(BenchmarkRunModel, identity) == getattr(run, identity))
        )
        if row is None:
            raise RuntimeError("Committed benchmark identity was not found")
        return self.to_entity(row)

    async def get_baseline(self, current: BenchmarkRun) -> BenchmarkRun | None:
        stmt = select(BenchmarkRunModel).where(
            BenchmarkRunModel.dataset == current.dataset,
            BenchmarkRunModel.llm_evaluated == current.llm_evaluated,
            BenchmarkRunModel.id != current.id,
            BenchmarkRunModel.creation_date <= current.creation_date,
        )
        # Missing legacy corpus metadata is compatible only with equally unknown metadata.
        mode = current.config_json.get("evaluation_mode", current.summary_metrics.get("evaluation_mode"))
        mode_expr = func.coalesce(
            BenchmarkRunModel.config_json["evaluation_mode"].as_string(),
            BenchmarkRunModel.summary_metrics["evaluation_mode"].as_string(),
        )
        stmt = stmt.where(mode_expr.is_(None) if mode is None else mode_expr == mode)
        for key in ("collection_name", "qdrant_url", "embedding_identity", "dataset_hash"):
            value = current.config_json.get(key, current.summary_metrics.get(key))
            expression = func.coalesce(
                BenchmarkRunModel.config_json[key].as_string(),
                BenchmarkRunModel.summary_metrics[key].as_string(),
            )
            stmt = stmt.where(expression.is_(None) if value is None else expression == str(value))
        row = await self._db.scalar(
            stmt.order_by(BenchmarkRunModel.creation_date.desc(), BenchmarkRunModel.id.desc()).limit(1)
        )
        return self.to_entity(row) if row else None

    async def export_header(self, run_id: int) -> dict | None:
        row = (
            (
                await self._db.execute(
                    select(
                        BenchmarkRunModel.id,
                        BenchmarkRunModel.dataset,
                        BenchmarkRunModel.config_json,
                        BenchmarkRunModel.summary_metrics,
                        BenchmarkRunModel.llm_evaluated,
                    ).where(BenchmarkRunModel.id == run_id)
                )
            )
            .mappings()
            .one_or_none()
        )
        return dict(row) if row else None

    async def result_page(self, run_id: int, *, offset: int, limit: int) -> list[dict]:
        # PostgreSQL expands the JSON array before LIMIT/OFFSET.
        column = BenchmarkRunModel.per_question_results
        query = (
            select(func.json_array_elements(column))
            .where(BenchmarkRunModel.id == run_id, func.json_typeof(column) == "array")
            .offset(offset)
            .limit(limit)
        )
        return list((await self._db.scalars(query)).all())

    async def set_artifact(self, run_id: int, report_format: str, metadata: dict) -> None:
        artifacts = await self._db.scalar(
            select(BenchmarkRunModel.artifacts).where(BenchmarkRunModel.id == run_id).with_for_update()
        )
        if artifacts is None:
            raise ValueError("Benchmark result not found")
        await self._db.execute(
            update(BenchmarkRunModel)
            .where(BenchmarkRunModel.id == run_id)
            .values(artifacts={**artifacts, report_format: metadata})
        )

    async def get_sweep_run_ids(self, sweep_id: int, limit: int) -> list[int]:
        query = (
            select(BenchmarkRunModel.id)
            .where(BenchmarkRunModel.sweep_id == sweep_id)
            .order_by(BenchmarkRunModel.id)
            .limit(limit)
        )
        return list((await self._db.scalars(query)).all())

    async def save_for_sweep(self, run: BenchmarkRun) -> BenchmarkRun:
        if run.sweep_id is None:
            raise ValueError("sweep_id is required")
        # The parent row serializes config lookup + insert across worker retries.
        await self._db.execute(
            select(BenchmarkSweepModel.id).where(BenchmarkSweepModel.id == run.sweep_id).with_for_update()
        )
        rows = await self._db.execute(
            select(BenchmarkRunModel).where(BenchmarkRunModel.sweep_id == run.sweep_id)
        )
        existing = next((row for row in rows.scalars() if row.config_json == run.config_json), None)
        if existing is None:
            return await self.create(run)
        existing.summary_metrics = run.summary_metrics
        existing.per_question_results = run.per_question_results
        existing.llm_evaluated = run.llm_evaluated
        existing.duration_sec = run.duration_sec
        await self._db.flush()
        return self.to_entity(existing)

    async def list_items(
        self,
        *,
        sweep_id: int | None = None,
        dataset: str | None = None,
        sort_by: str = "creation_date",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> list[BenchmarkRun]:
        sort_col = getattr(BenchmarkRunModel, sort_by, BenchmarkRunModel.creation_date)
        if sort_order == "asc":
            stmt = select(BenchmarkRunModel).order_by(sort_col.asc())
        else:
            stmt = select(BenchmarkRunModel).order_by(sort_col.desc())

        if sweep_id is not None:
            stmt = stmt.where(BenchmarkRunModel.sweep_id == sweep_id)
        if dataset is not None:
            stmt = stmt.where(BenchmarkRunModel.dataset == dataset)

        stmt = stmt.offset(offset).limit(limit)
        result = await self._db.execute(stmt)
        return [self.to_entity(orm) for orm in result.scalars().all()]

    async def count(
        self,
        *,
        sweep_id: int | None = None,
        dataset: str | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(BenchmarkRunModel)
        if sweep_id is not None:
            stmt = stmt.where(BenchmarkRunModel.sweep_id == sweep_id)
        if dataset is not None:
            stmt = stmt.where(BenchmarkRunModel.dataset == dataset)
        result = await self._db.execute(stmt)
        return result.scalar_one()

    async def get_by_ids(self, ids: list[int]) -> list[BenchmarkRun]:
        if not ids:
            return []
        stmt = (
            select(BenchmarkRunModel)
            .where(BenchmarkRunModel.id.in_(ids))
            .order_by(BenchmarkRunModel.creation_date.desc())
        )
        result = await self._db.execute(stmt)
        return [self.to_entity(orm) for orm in result.scalars().all()]

    async def get_latest(self, dataset: str | None = None) -> BenchmarkRun | None:
        stmt = select(BenchmarkRunModel).order_by(BenchmarkRunModel.creation_date.desc())
        if dataset is not None:
            stmt = stmt.where(BenchmarkRunModel.dataset == dataset)
        stmt = stmt.limit(1)
        result = await self._db.execute(stmt)
        orm = result.scalar_one_or_none()
        return self.to_entity(orm) if orm else None

    @staticmethod
    def to_entity(orm: BenchmarkRunModel) -> BenchmarkRun:
        summary = dict(orm.summary_metrics or {})
        # Older sweep summaries omitted answer averages, but retained question scores.
        missing = [metric for metric in EVIDENCE_METRICS if summary.get(f"avg_{metric}") is None]
        if missing and orm.per_question_results:
            recovered = summarize_evidence(orm.per_question_results)
            for metric in missing:
                average = recovered[f"avg_{metric}"]
                if average is not None:
                    summary[f"avg_{metric}"] = average
                    summary[f"{metric}_evaluated_count"] = recovered[f"{metric}_evaluated_count"]
        return BenchmarkRun(
            id=orm.id,
            creation_date=orm.creation_date,
            sweep_id=orm.sweep_id,
            config_json=orm.config_json or {},
            summary_metrics=summary,
            duration_sec=orm.duration_sec,
            llm_evaluated=orm.llm_evaluated,
            dataset=orm.dataset,
            per_question_results=orm.per_question_results,
            filename=orm.filename,
            job_id=orm.job_id,
            legacy_hash=orm.legacy_hash,
            artifacts=orm.artifacts or {},
        )
