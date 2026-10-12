"""Saved answer coverage remains available across history selections."""

from unittest.mock import AsyncMock, Mock

import pytest

from infrastructure.database.models import BenchmarkRunModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_run_repository import (
    SQLAlchemyBenchmarkRunRepository,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", [{}, {"avg_answer_condition_coverage": None}])
async def test_history_recovers_missing_answer_coverage_without_changing_stored_data(stored):
    row = BenchmarkRunModel(
        id=12,
        summary_metrics=stored,
        per_question_results=[
            {"evidence_metrics": {"answer_condition_coverage": value}} for value in [0, 1, 1, 1, 1, 0, None]
        ],
    )
    db = AsyncMock()
    db.execute.return_value = Mock(scalar_one_or_none=Mock(return_value=row))
    repository = SQLAlchemyBenchmarkRunRepository(db)

    run = await repository.get_by_id(12)

    assert run.summary_metrics["avg_answer_condition_coverage"] == 0.6667
    assert run.summary_metrics["answer_condition_coverage_evaluated_count"] == 6
    assert row.summary_metrics == stored
    assert "answer_condition_coverage_evaluated_count" not in row.summary_metrics


@pytest.mark.asyncio
async def test_history_preserves_saved_averages_and_unmeasured_metrics():
    row = BenchmarkRunModel(
        id=12,
        summary_metrics={"avg_answer_condition_coverage": 0.75},
        per_question_results=[{"evidence_metrics": {"answer_condition_coverage": 0}}],
    )
    db = AsyncMock()
    db.execute.return_value = Mock(scalar_one_or_none=Mock(return_value=row))

    run = await SQLAlchemyBenchmarkRunRepository(db).get_by_id(12)

    assert run.summary_metrics["avg_answer_condition_coverage"] == 0.75
    assert run.summary_metrics.get("avg_answer_fact_coverage") is None
