"""Public imports and runner forwarding after the benchmark feature move."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from application.benchmark import BenchmarkService


if TYPE_CHECKING:
    from application.benchmark import BenchmarkRunOptions

LEGACY_EXPORTS = [
    (
        "application.dto.benchmark_dto",
        (
            "ApplyConfigResult",
            "BenchmarkQuestionCreateDTO",
            "SweepCreateDTO",
            "RegressionCheckOutput",
            "RegressionCheckResult",
        ),
    ),
    ("application.ports.benchmark_checkpoints", ("BenchmarkCheckpoints",)),
    ("application.ports.benchmark_history", ("BenchmarkHistoryPort",)),
    ("application.ports.benchmark_runner", ("BenchmarkRunnerPort",)),
    ("application.services.benchmark_dataset", ("load_benchmark_questions",)),
    (
        "application.services.benchmark_export_service",
        ("BenchmarkExportService", "CSV_FIELDS", "EXPORT_CHUNK_BYTES", "encode_json", "encode_csv"),
    ),
    ("application.services.benchmark_import_service", ("BenchmarkImportService", "BenchmarkImportResult")),
    (
        "application.services.benchmark_orchestrator",
        ("BenchmarkService", "compute_summary_from_results"),
    ),
    (
        "application.services.benchmark_result_service",
        (
            "BenchmarkResultService",
            "BenchmarkResultSummary",
            "BenchmarkResultDetail",
            "BenchmarkResultsList",
            "normalize_metrics",
        ),
    ),
    (
        "application.services.benchmark_services",
        (
            "BenchmarkQuestionService",
            "BenchmarkRunService",
            "BenchmarkSweepService",
            "RunCompareResult",
            "RUN_CONFIG_KEY_MAP",
            "HISTORY_CONFIG_KEYS",
        ),
    ),
]


@pytest.mark.parametrize("legacy,names", LEGACY_EXPORTS)
def test_legacy_exports_share_canonical_objects(legacy: str, names: tuple[str, ...]) -> None:
    old = importlib.import_module(legacy)
    canonical = importlib.import_module("application.benchmark")

    for name in names:
        assert getattr(old, name) is getattr(canonical, name)


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_options", [False, True])
async def test_runner_receives_defaults_and_resume_options(*, explicit_options: bool) -> None:
    runner = AsyncMock()
    runner.run.return_value = []
    rag = AsyncMock()
    checkpoints = AsyncMock()
    options: BenchmarkRunOptions = {}

    if explicit_options:
        options = {
            "seed": 42,
            "n_runs": 2,
            "max_concurrent": 3,
            "resume": True,
            "checkpoints": checkpoints,
            "checkpoint_prefix": "job:7",
        }

    service = BenchmarkService(runner=runner)
    service.set_rag_service(rag)
    summary = await service.run([], None, 4, "judge", **options)
    runner.run.assert_awaited_once_with(
        questions=[],
        out_dir=None,
        top_k=4,
        judge_model="judge",
        seed=42 if explicit_options else None,
        n_runs=2 if explicit_options else 1,
        max_concurrent=3 if explicit_options else None,
        rag_service=rag,
        resume=explicit_options,
        checkpoints=checkpoints if explicit_options else None,
        checkpoint_prefix="job:7" if explicit_options else None,
        export_files=False,
    )
    assert summary["status"] == "done"
    assert summary["total_questions"] == 0
    assert summary["results"] == []


@pytest.mark.asyncio
async def test_runner_must_be_injected() -> None:
    with pytest.raises(RuntimeError, match="A benchmark runner must be injected"):
        await BenchmarkService().run([], None, 4, "judge")
