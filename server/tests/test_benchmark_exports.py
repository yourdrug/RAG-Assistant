"""Bounded DB export, artifact retries, legacy validation and admin boundary."""

from __future__ import annotations

from contextlib import asynccontextmanager
import csv
from functools import partial
import io
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

from fastapi import FastAPI, HTTPException
import httpx
import pytest
from typer.testing import CliRunner

from application.benchmark import (
    EXPORT_CHUNK_BYTES,
    BenchmarkExportService,
    BenchmarkImportService,
    BenchmarkResultService,
    jobs as benchmark_jobs,
)
from application.benchmark.jobs import BenchmarkJobService
from application.ports.route_configs import BenchmarkConfig
from composition.cli.cli import cli
from composition.cli.commands import benchmark as command
from composition.service_providers import build_sweep_engine
from config import settings
from domain.shared.exceptions import PermissionDeniedError
from infrastructure.benchmark.corpus import benchmark_scope
from infrastructure.benchmark.reporting.legacy_import import load_legacy_runs
from infrastructure.benchmark.sweep.settings import LiveSweepSettings
from infrastructure.database.database import database
from infrastructure.worker import runtime, tasks
from presentation.api.auth_dependencies import get_current_user
from presentation.api.dependencies import IdempotencyContext, JobSubmission, create_benchmark_export_service
from presentation.api.routes import benchmark
from presentation.api.schemas.benchmark import BenchmarkRequest
from tests.fakes import FakeUnitOfWorkFactory, require_present
from tests.test_benchmark_characterization import _result


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from starlette.requests import Request

    from application.jobs.service import JobService
    from application.uow.factory import UnitOfWorkFactory
    from domain.benchmark.run import BenchmarkRun
    from presentation.api.schemas.auth import CurrentUser


class Factory:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.open = 0
        self.saved: list[Any] = []
        self.repo = SimpleNamespace(
            export_header=AsyncMock(return_value={"id": 7, "dataset": "legal"}),
            result_page=AsyncMock(side_effect=lambda _run_id, offset, limit: rows[offset : offset + limit]),
            set_artifact=AsyncMock(side_effect=lambda *args: self.saved.append(args)),
        )

    @asynccontextmanager
    async def create(self, **kwargs: Any) -> AsyncIterator[SimpleNamespace]:
        _ = kwargs
        self.open += 1
        try:
            yield SimpleNamespace(benchmark_runs=self.repo)
        finally:
            self.open -= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("export_format", ["json", "csv"])
async def test_export_without_data_dir_and_without_open_transaction(
    export_format: str, monkeypatch: pytest.MonkeyPatch
) -> None:

    monkeypatch.setattr(settings, "data_dir", "/proc/readonly-unavailable")
    rows = [{"id": 1, "question": "=1+1", "answer": 'Привет, "мир"\nline', "faithfulness": 8}]
    factory = Factory(rows)
    blocks = []

    async for block in BenchmarkExportService(cast("UnitOfWorkFactory", factory)).export(7, export_format):
        assert factory.open == 0
        blocks.append(block)

    data = b"".join(blocks).decode()

    if export_format == "json":
        assert json.loads(data)["results"] == rows
    else:
        row = next(iter(csv.DictReader(io.StringIO(data))))
        assert row["question"] == "'=1+1"
        assert row["answer"] == rows[0]["answer"]


@pytest.mark.asyncio
async def test_large_export_pages_and_chunks_are_bounded() -> None:
    factory = Factory([{"id": n, "answer": "я" * 200_000} for n in range(30)])
    count = 0

    async for block in BenchmarkExportService(cast("UnitOfWorkFactory", factory)).export(7, "json"):
        assert len(block) <= EXPORT_CHUNK_BYTES
        count += len(block)

    assert count > 10_000_000
    assert all(call.kwargs["limit"] == 1 for call in factory.repo.result_page.call_args_list)


@pytest.mark.asyncio
async def test_artifact_failure_retry_preserves_result_and_key() -> None:
    factory = Factory([{"id": 1}])
    keys = []
    failed = True

    async def upload(key: str, blocks: AsyncIterator[bytes]) -> None:
        keys.append(key)

        async for _ in blocks:
            assert factory.open == 0

        if failed:
            msg = "storage unavailable"
            raise OSError(msg)

    service = BenchmarkExportService(
        cast("UnitOfWorkFactory", factory), SimpleNamespace(upload_stream=upload)
    )
    assert await service.save_report(7, "json") is False
    assert factory.saved[-1][2]["status"] == "failed"
    failed = False
    assert await service.save_report(7, "json") is True
    assert factory.saved[-1][2]["size_bytes"] > 0
    assert keys == ["benchmarks/runs/7/report.json"] * 2


def test_partial_and_malformed_legacy_records_keep_files(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps({"timestamp": "2025-01-01", "metrics": {"hit_rate": 0.8}}) + "\n{broken}\n[]\n"
    )
    runs, errors = load_legacy_runs(path, "legacy")
    assert len(runs) == 1
    assert len(errors) == 2
    assert runs[0].per_question_results is None
    assert runs[0].config_json["legacy_import"]["sha256"]
    assert path.exists()


@pytest.mark.asyncio
async def test_legacy_dry_run_and_repeat_do_not_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps({"timestamp": "2025-01-01", "metrics": {"hit_rate": 0.8}}))
    runs, _ = load_legacy_runs(path, "legacy")
    saved: dict[str, Any] = {}

    async def get(digest: str) -> Any:
        return saved.get(digest)

    async def save(run: BenchmarkRun) -> Any:
        saved.setdefault(require_present(run.legacy_hash), run)
        return saved[require_present(run.legacy_hash)]

    factory = Factory([])
    factory.repo.get_by_legacy_hash = AsyncMock(side_effect=get)
    factory.repo.save_once = AsyncMock(side_effect=save)
    service = BenchmarkImportService(cast("UnitOfWorkFactory", factory))
    assert (await service.import_runs(runs)).imported == 1
    assert not saved
    assert (await service.import_runs(runs, apply=True)).imported == 1
    assert (await service.import_runs(runs, apply=True)).duplicates == 1
    assert len(saved) == 1


@pytest.mark.asyncio
async def test_export_requires_admin_and_streams_authorized_download() -> None:
    app = FastAPI()
    app.include_router(benchmark.router)
    service = BenchmarkExportService(cast("UnitOfWorkFactory", Factory([{"id": 1}])))
    app.dependency_overrides[create_benchmark_export_service] = lambda: service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:  # type: ignore[arg-type]  # Starlette and HTTPX disagree on the ASGI send annotation.
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, role="user")
        with pytest.raises(PermissionDeniedError):
            await client.get("/benchmark/results/7/export")
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, role="admin")
        response = await client.get("/benchmark/results/7/export?format=json")
        assert response.status_code == 200
        assert response.json()["results"] == [{"id": 1}]
        assert "attachment" in response.headers["content-disposition"]
        assert (await client.get("/benchmark/results/7/export?format=html")).status_code == 422


@pytest.mark.asyncio
async def test_legacy_out_dir_is_noop_only_for_previous_default() -> None:

    request = SimpleNamespace(method="POST", url=SimpleNamespace(path="/benchmark"))
    user = SimpleNamespace(kind="internal", role="admin", id=1)
    cfg = BenchmarkConfig(data_dir="/proc/readonly", retriever_top_k=4, llm_model="judge")
    jobs = SimpleNamespace(create_job=AsyncMock(return_value=8))
    enqueuer = SimpleNamespace(enqueue_benchmark=AsyncMock())
    # Empty idempotency headers bypass the reservation store.
    await benchmark.run_benchmark(
        request=cast("Request", request),
        req=BenchmarkRequest(out_dir="/proc/readonly/benchmark_results"),
        admin=cast("CurrentUser", user),
        bench_cfg=cfg,
        submission=JobSubmission(jobs=cast("JobService", jobs), enqueuer=enqueuer, log=Mock()),
        idempotency=IdempotencyContext(key=None, store=MagicMock()),
    )
    assert "out_dir" not in enqueuer.enqueue_benchmark.call_args.kwargs
    with pytest.raises(HTTPException) as exc:
        await benchmark.run_benchmark(
            request=cast("Request", request),
            req=BenchmarkRequest(out_dir="legacy-output/custom"),
            admin=cast("CurrentUser", user),
            bench_cfg=cfg,
            submission=JobSubmission(jobs=cast("JobService", jobs), enqueuer=enqueuer, log=Mock()),
            idempotency=IdempotencyContext(key=None, store=MagicMock()),
        )
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_worker_retry_reuses_committed_run(monkeypatch: pytest.MonkeyPatch) -> None:

    factory = Factory([])
    saved = None
    evaluations = AsyncMock(return_value={"results": [{"id": 1}], "total_time_sec": 1})

    async def get_job(job_id: object) -> Any:
        return saved

    async def save(run: BenchmarkRun) -> Any:
        nonlocal saved
        run.id = 7
        saved = run
        return run

    async def tracked(uow: object, job_id: object, action: Any, **kwargs: Any) -> None:
        await action()

    factory.repo.get_by_job_id = AsyncMock(side_effect=get_job)
    factory.repo.save_once = AsyncMock(side_effect=save)
    monkeypatch.setattr(
        benchmark_jobs, "load_benchmark_questions", AsyncMock(return_value=[{"question": "q"}])
    )
    monkeypatch.setattr(runtime, "run_tracked_job", tracked)
    reports = SimpleNamespace(save_report=AsyncMock(return_value=False))

    ctx: dict[str, Any] = {
        "benchmark_results": BenchmarkResultService(cast("UnitOfWorkFactory", factory)),
        "benchmark_exports": reports,
        "container": SimpleNamespace(
            infrastructure=SimpleNamespace(
                db=SimpleNamespace(uow_factory=factory),
                ml=SimpleNamespace(
                    ml_clients=None, file_storage=object(), benchmark_service=SimpleNamespace(run=evaluations)
                ),
            ),
            application=SimpleNamespace(rag_service=object()),
        ),
    }
    ctx["uow_factory"] = factory
    ctx["benchmark_job"] = BenchmarkJobService(
        cast("UnitOfWorkFactory", factory),
        ctx["container"].infrastructure.ml.benchmark_service,
        cast("BenchmarkResultService", ctx["benchmark_results"]),
        cast("BenchmarkExportService", reports),
        partial(benchmark_scope, None),
    )
    del ctx["container"]

    for _ in range(2):
        await tasks.run_benchmark(ctx, dataset="main", top_k=4, judge_model="judge", job_id=8)

    evaluations.assert_awaited_once()
    factory.repo.save_once.assert_awaited_once()
    assert require_present(saved).job_id == 8
    assert reports.save_report.await_count == 4


def test_cli_registry_has_explicit_export_and_requires_out() -> None:

    runner = CliRunner()
    response = runner.invoke(cli.cli, ["benchmark", "export", "--help"])
    assert response.exit_code == 0, response.output
    assert "--out" in response.output
    assert runner.invoke(cli.cli, ["benchmark", "export", "7"]).exit_code != 0


@pytest.mark.asyncio
async def test_cli_export_creates_only_explicit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    factory = Factory([{"id": 1, "answer": "ok"}])
    monkeypatch.setattr(command, "UnitOfWorkFactory", lambda _database: factory)
    monkeypatch.setattr(database, "connect", AsyncMock())
    monkeypatch.setattr(database, "disconnect", AsyncMock())
    out = tmp_path / "chosen.json"
    await command.export_run(7, "json", out)
    assert json.loads(out.read_bytes())["results"] == [{"id": 1, "answer": "ok"}]
    assert list(tmp_path.iterdir()) == [out]  # noqa: ASYNC240 - Small local fixture I/O in the test setup.
    with pytest.raises(FileExistsError):
        await command.export_run(7, "json", out)


def test_legacy_json_report_preserves_per_question_results(tmp_path: Path) -> None:

    path = tmp_path / "report.json"
    rows = [_result()]
    path.write_text(json.dumps(rows))
    runs, errors = load_legacy_runs(path, "legacy")
    assert not errors
    assert runs[0].per_question_results == rows
    assert runs[0].summary_metrics["avg_faithfulness"] == 8


@pytest.mark.asyncio
async def test_default_web_sweep_does_not_fallback_to_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(settings, "data_dir", "/proc/unavailable")
    assert not hasattr(LiveSweepSettings(), "results_path")
    with pytest.raises(RuntimeError, match="database checkpoint"):
        build_sweep_engine(FakeUnitOfWorkFactory()).create_checkpoints(7)
