"""Public imports and runner forwarding after the benchmark feature move."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from application.benchmark import BenchmarkService


if TYPE_CHECKING:
    from application.benchmark import BenchmarkRunOptions


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.benchmark", "application.benchmark.dto", "ApplyConfigResult"),
        ("application.benchmark", "application.benchmark.dto", "BenchmarkQuestionCreateDTO"),
        ("application.benchmark", "application.benchmark.dto", "SweepCreateDTO"),
        ("application.benchmark", "application.benchmark.dto", "RegressionCheckOutput"),
        ("application.benchmark", "application.benchmark.dto", "RegressionCheckResult"),
        ("application.benchmark", "application.benchmark.ports", "BenchmarkCheckpoints"),
        ("application.benchmark", "application.benchmark.ports", "BenchmarkHistoryPort"),
        ("application.benchmark", "application.benchmark.ports", "BenchmarkRunnerPort"),
        ("application.benchmark", "application.benchmark.dataset", "load_benchmark_questions"),
        ("application.benchmark", "application.benchmark.export_service", "BenchmarkExportService"),
        ("application.benchmark", "application.benchmark.export_service", "CSV_FIELDS"),
        ("application.benchmark", "application.benchmark.export_service", "EXPORT_CHUNK_BYTES"),
        ("application.benchmark", "application.benchmark.export_service", "encode_json"),
        ("application.benchmark", "application.benchmark.export_service", "encode_csv"),
        ("application.benchmark", "application.benchmark.import_service", "BenchmarkImportService"),
        ("application.benchmark", "application.benchmark.dto", "BenchmarkImportResult"),
        ("application.benchmark", "application.benchmark.orchestrator", "BenchmarkService"),
        ("application.benchmark", "application.benchmark.orchestrator", "compute_summary_from_results"),
        ("application.benchmark", "application.benchmark.result_service", "BenchmarkResultService"),
        ("application.benchmark", "application.benchmark.dto", "BenchmarkResultSummary"),
        ("application.benchmark", "application.benchmark.dto", "BenchmarkResultDetail"),
        ("application.benchmark", "application.benchmark.dto", "BenchmarkResultsList"),
        ("application.benchmark", "application.benchmark.result_service", "normalize_metrics"),
        ("application.benchmark", "application.benchmark.services", "BenchmarkQuestionService"),
        ("application.benchmark", "application.benchmark.services", "BenchmarkRunService"),
        ("application.benchmark", "application.benchmark.services", "BenchmarkSweepService"),
        ("application.benchmark", "application.benchmark.dto", "RunCompareResult"),
        ("application.benchmark", "application.benchmark.services", "RUN_CONFIG_KEY_MAP"),
        ("application.benchmark", "application.benchmark.services", "HISTORY_CONFIG_KEYS"),
    ],
)
def test_public_exports_share_canonical_objects(public: str, implementation: str, name: str) -> None:
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )


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
