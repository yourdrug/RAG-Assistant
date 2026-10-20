"""Database history, compatible baselines and retry identity."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from application.benchmark import BenchmarkResultService
from domain.benchmark.run import BenchmarkRun
from infrastructure.benchmark.benchmark_history_adapter import BenchmarkHistoryAdapter


class Factory:
    def __init__(self, repo):
        self.repo = repo

    @asynccontextmanager
    async def create(self, **kwargs):
        yield SimpleNamespace(benchmark_runs=self.repo)


@pytest.mark.asyncio
async def test_regression_uses_db_baseline_and_exposes_identity():
    current = BenchmarkRun(id=3, dataset='legal', summary_metrics={'hit_rate': 0.7})
    baseline = BenchmarkRun(id=1, dataset='legal', summary_metrics={'hit_rate': 0.9})
    repo = SimpleNamespace(
        get_by_id=AsyncMock(return_value=current), get_baseline=AsyncMock(return_value=baseline)
    )
    output = await BenchmarkResultService(Factory(repo)).check_regression(3, BenchmarkHistoryAdapter())
    assert output.passed is False
    assert output.run_id == 3
    assert output.baseline_run_id == 1
    repo.get_baseline.assert_awaited_once_with(current)


@pytest.mark.asyncio
async def test_missing_baseline_is_explicit():
    repo = SimpleNamespace(
        get_latest=AsyncMock(return_value=BenchmarkRun(id=8)), get_baseline=AsyncMock(return_value=None)
    )
    output = await BenchmarkResultService(Factory(repo)).check_regression(None, BenchmarkHistoryAdapter())
    assert output.baseline_run_id is None
    assert output.run_id == 8
    assert output.results == []
