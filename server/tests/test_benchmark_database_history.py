"""Database history, compatible baselines and retry identity."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock

import pytest

from application.benchmark import BenchmarkResultService, RegressionCheckOutput
from composition.cli.commands import benchmark as benchmark_cli
from composition.domain_packs import PACK_REGISTRATIONS
from domain.benchmark.policy import GateStatus
from domain.benchmark.run import BenchmarkRun
from infrastructure.benchmark.reporting.benchmark_history_adapter import BenchmarkHistoryAdapter


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from application.uow.factory import UnitOfWorkFactory


class Factory:
    def __init__(self, repo: Any) -> None:
        self.repo = repo

    @asynccontextmanager
    async def create(self, **kwargs: Any) -> AsyncIterator[SimpleNamespace]:
        del kwargs
        yield SimpleNamespace(
            benchmark_runs=self.repo,
            benchmark_quality=SimpleNamespace(get_policy=AsyncMock(return_value=None)),
        )


@pytest.mark.asyncio
async def test_regression_uses_db_baseline_and_exposes_identity() -> None:
    current = BenchmarkRun(id=3, dataset="legal", summary_metrics={"hit_rate": 0.7})
    baseline = BenchmarkRun(id=1, dataset="legal", summary_metrics={"hit_rate": 0.9})
    repo = SimpleNamespace(
        get_by_id=AsyncMock(return_value=current), get_baseline=AsyncMock(return_value=baseline)
    )
    output = await BenchmarkResultService(cast("UnitOfWorkFactory", Factory(repo))).check_regression(
        3, BenchmarkHistoryAdapter()
    )
    assert output.passed is False
    assert output.run_id == 3
    assert output.baseline_run_id == 1
    repo.get_baseline.assert_awaited_once_with(current)


@pytest.mark.asyncio
async def test_missing_baseline_is_explicit() -> None:
    repo = SimpleNamespace(
        get_latest=AsyncMock(return_value=BenchmarkRun(id=8)), get_baseline=AsyncMock(return_value=None)
    )
    output = await BenchmarkResultService(cast("UnitOfWorkFactory", Factory(repo))).check_regression(
        None, BenchmarkHistoryAdapter()
    )
    assert output.baseline_run_id is None
    assert output.run_id == 8
    assert output.results == []
    assert output.status is GateStatus.INSUFFICIENT_DATA
    assert not output.passed


@pytest.mark.asyncio
async def test_missing_run_is_insufficient() -> None:
    repo = SimpleNamespace(get_latest=AsyncMock(return_value=None))
    output = await BenchmarkResultService(cast("UnitOfWorkFactory", Factory(repo))).check_regression(
        None, BenchmarkHistoryAdapter(), dataset="support-main"
    )
    assert output.status is GateStatus.INSUFFICIENT_DATA
    assert not output.passed


def test_gate_checks_quick_set_of_every_pack_and_never_passes_without_data() -> None:
    assert benchmark_cli.gate_datasets() == [f"{pack}-quick" for pack in PACK_REGISTRATIONS]

    def output(status: GateStatus) -> RegressionCheckOutput:
        return RegressionCheckOutput(passed=status is GateStatus.PASSED, status=status)

    passed, failed, missing = (output(s) for s in GateStatus)
    assert benchmark_cli.overall_status([passed, passed]) is GateStatus.PASSED
    assert benchmark_cli.overall_status([passed, missing]) is GateStatus.INSUFFICIENT_DATA
    assert benchmark_cli.overall_status([missing, failed]) is GateStatus.FAILED
    assert benchmark_cli.overall_status([]) is GateStatus.INSUFFICIENT_DATA
