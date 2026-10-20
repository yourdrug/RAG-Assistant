"""Enum normalization preserves stored and public string values."""

from collections.abc import Callable
from enum import StrEnum
import json
from typing import Any

import pytest

from domain.benchmark.sweep import (
    BenchmarkStrategy,
    BenchmarkSweep,
    BenchmarkSweepStatus,
    SweepEvaluationMode,
)
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.runtime.jobs import BackgroundJob, BackgroundJobStatus
from infrastructure.database.models import BackgroundJobModel, BenchmarkSweepModel
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_sweep_repository import (
    SQLAlchemyBenchmarkSweepRepository,
)
from infrastructure.repositories.misc.sqlalchemy_background_job_repository import (
    SQLAlchemyBackgroundJobRepository,
)


@pytest.mark.parametrize(
    "factory, field, enum_type",
    [
        (BackgroundJob, "status", BackgroundJobStatus),
        (BenchmarkSweep, "status", BenchmarkSweepStatus),
        (BenchmarkSweep, "strategy", BenchmarkStrategy),
        (BenchmarkSweep, "evaluation_mode", SweepEvaluationMode),
        (lambda **kw: UserContext(user_id=1, user_role=UserRole.USER, **kw), "user_kind", UserKind),
        (lambda **kw: UserContext(user_id=1, user_kind=UserKind.INTERNAL, **kw), "user_role", UserRole),
    ],
)
def test_enum_fields_normalize_and_reject_unknown_values(
    factory: Callable[..., Any], field: str, enum_type: type[StrEnum]
) -> None:
    for member in enum_type:
        for value in (member, member.value):
            entity = factory(**{field: value})
            assert getattr(entity, field) is member
            assert json.loads(json.dumps({field: getattr(entity, field)}))[field] == member.value

    with pytest.raises(ValueError, match="unknown"):
        factory(**{field: "unknown"})


@pytest.mark.parametrize(
    "transition, expected",
    [
        ("mark_running", BackgroundJobStatus.RUNNING),
        ("mark_done", BackgroundJobStatus.DONE),
        ("mark_failed", BackgroundJobStatus.FAILED),
    ],
)
def test_job_transitions_keep_enum_status(transition: str, expected: BackgroundJobStatus) -> None:
    job = BackgroundJob()
    assert isinstance(job.status, BackgroundJobStatus)
    getattr(job, transition)()
    assert job.status is expected


def test_default_sweep_fields_are_enums() -> None:
    sweep = BenchmarkSweep()
    assert sweep.status is BenchmarkSweepStatus.PENDING
    assert sweep.strategy is BenchmarkStrategy.GRID
    assert sweep.evaluation_mode is SweepEvaluationMode.FAST


def test_orm_strings_become_enum_fields() -> None:
    job = SQLAlchemyBackgroundJobRepository._to_entity(
        BackgroundJobModel(id=1, job_type="benchmark", status="running")
    )
    assert job.status is BackgroundJobStatus.RUNNING
    sweep = SQLAlchemyBenchmarkSweepRepository.to_entity(
        BenchmarkSweepModel(id=1, status="cancelled", strategy="grid", evaluation_mode="full")
    )
    assert sweep.status is BenchmarkSweepStatus.CANCELLED
    assert sweep.strategy is BenchmarkStrategy.GRID
    assert sweep.evaluation_mode is SweepEvaluationMode.FULL
