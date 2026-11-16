"""Shared worker policy preserves the job principal and processing retry boundaries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

from arq import Retry
import pytest

from application.jobs.principal import job_principal
from config import settings
from domain.runtime.jobs import BackgroundJob, JobType
from infrastructure.metrics import metrics
from infrastructure.redis.redis_client import redis_client
from infrastructure.worker import runtime, sweep, tasks
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from application.documents.dto import ProcessDocumentCommand


class JobFactory:
    def __init__(self) -> None:
        self.jobs = SimpleNamespace(
            mark_running=AsyncMock(),
            mark_done=AsyncMock(),
            mark_failed=AsyncMock(),
            touch_heartbeat=AsyncMock(),
        )
        self.attempts: dict[object, int] = {}

        async def claim(*args: object, **kwargs: object) -> int:
            self.attempts[args[0]] = self.attempts.get(args[0], 0) + 1
            await self.jobs.mark_running(args[0])
            return self.attempts[args[0]]

        async def finish(
            job_id: int, attempt_id: str, *, error: str | None = None, retry_seconds: float | None = None
        ) -> bool:
            if retry_seconds is not None:
                return True

            if error:
                await self.jobs.mark_failed(job_id, error)
            else:
                await self.jobs.mark_done(job_id)

            return True

        self.jobs.claim = claim
        self.jobs.finish_attempt = finish
        self.jobs.renew_lease = AsyncMock(return_value=True)

    def create(self, *, master: bool = False) -> JobFactory:
        assert master
        return self

    async def __aenter__(self) -> SimpleNamespace:
        return SimpleNamespace(background_jobs=self.jobs)

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_ten_documents_from_same_principal_start_together(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "document_max_concurrent", 10)
    factory = JobFactory()
    started: set[int] = set()
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def process(command: ProcessDocumentCommand) -> None:
        started.add(command.document_id)
        assert job_principal.get() == "user:9"

        if len(started) == 10:
            all_started.set()

        await release.wait()

    processor = SimpleNamespace(process=process)
    ctx = {"uow_factory": factory, "document_processor": processor}
    batch = [
        asyncio.create_task(
            tasks.process_document(
                ctx,
                document_id=document_id,
                storage_key=f"uploads/{document_id}.txt",
                filename=f"{document_id}.txt",
                visibility="internal_public",
                owner_id=9,
                principal_id=9,
                group_id=None,
                replace_id=None,
                job_id=document_id,
                cost_principal="user:9",
            )
        )
        for document_id in range(1, 11)
    ]
    try:
        await asyncio.wait_for(all_started.wait(), timeout=5)
        assert factory.jobs.mark_running.await_count == 10
        factory.jobs.mark_done.assert_not_awaited()
    finally:
        release.set()
        await asyncio.gather(*batch)
    assert started == set(range(1, 11))
    assert factory.jobs.mark_done.await_count == 10
    assert job_principal.get() is None


@pytest.mark.asyncio
async def test_processing_retries_preserve_principal() -> None:
    factory = JobFactory()
    observed = []

    @runtime.job(JobType.DOCUMENT_PROCESSING)
    async def action(ctx: dict[str, int], **kwargs: object) -> None:
        observed.append((ctx["job_try"], job_principal.get()))
        message = "failed"
        raise ConnectionError(message)

    ctx = {"uow_factory": factory}
    arguments = {"job_id": 5, "principal_id": 9, "cost_principal": "user:9"}

    for attempt in (1, 2):
        with pytest.raises(Retry) as raised:
            await action(dict(ctx, job_try=attempt), **arguments)
        assert 5000 * 2 ** (attempt - 1) <= raised.value.defer_score <= 6000 * 2 ** (attempt - 1)

    await action(dict(ctx, job_try=3), **arguments)
    assert [row[0] for row in observed] == [1, 2, 3]
    assert all(row[1] == "user:9" for row in observed)
    factory.jobs.mark_failed.assert_awaited_once_with(5, "failed")
    factory.jobs.mark_done.assert_not_awaited()
    assert job_principal.get() is None


@pytest.mark.asyncio
async def test_cancellation_releases_principal_context() -> None:
    factory = JobFactory()

    @runtime.job(JobType.DOCUMENT_PROCESSING, max_tries=1)
    async def action(ctx: dict[str, int], **kwargs: object) -> None:
        raise asyncio.CancelledError

    ctx = {"uow_factory": factory}
    with pytest.raises(asyncio.CancelledError):
        await action(ctx, job_id=7, owner_id=8, cost_principal="user:8")
    factory.jobs.mark_failed.assert_awaited_once()
    assert job_principal.get() is None


@pytest.mark.asyncio
async def test_saturated_concurrency_group_defers_without_claiming(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ingest_max_concurrent", 1)
    factory = FakeUnitOfWorkFactory()
    repo = factory._uow.fake_background_jobs
    running = await repo.create(BackgroundJob(job_type=JobType.INGEST))
    waiting = await repo.create(BackgroundJob(job_type=JobType.INGEST_FILE))
    assert running.id is not None
    assert waiting.id is not None
    running.mark_running()
    ran: list[int] = []

    @runtime.job(JobType.INGEST_FILE)
    async def run_single_ingest(ctx: dict[str, object], **kwargs: object) -> None:
        ran.append(cast("int", kwargs["job_id"]))

    ctx: dict[str, object] = {"uow_factory": factory}

    with pytest.raises(Retry):
        await run_single_ingest(ctx, job_id=waiting.id)

    assert ran == []
    # The fake UoW restores a snapshot on rollback: read state from the repository.
    assert repo.jobs[waiting.id].status == "pending"
    assert repo.jobs[waiting.id].attempt == 0, "a deferred delivery must not consume an execution attempt"

    repo.jobs[running.id].mark_done()
    await run_single_ingest(ctx, job_id=waiting.id)
    assert ran == [waiting.id]
    assert repo.jobs[waiting.id].status == "done"


@pytest.mark.asyncio
async def test_infra_metrics_collect_database_and_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    db, qdrant = AsyncMock(), AsyncMock()
    monkeypatch.setattr(metrics, "collect_db_pool_metrics", db)
    monkeypatch.setattr(metrics, "collect_qdrant_metrics", qdrant)
    await metrics.collect_infra_metrics()
    db.assert_awaited_once()
    qdrant.assert_awaited_once()


@pytest.mark.asyncio
async def test_sweep_events_reuse_process_owned_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = SimpleNamespace(publish=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr(redis_client, "_redis", redis)
    await sweep.publish_sweep_event(4, {"evaluated": 1})
    await sweep.publish_sweep_event(4, {"done": True})
    assert redis.publish.await_count == 2
    assert redis.publish.await_args.args == ("sweep:4", '{"done": true}')
    redis.aclose.assert_not_awaited()
