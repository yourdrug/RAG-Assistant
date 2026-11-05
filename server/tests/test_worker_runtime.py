"""Shared worker policy preserves billing and processing retry boundaries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from arq import Retry
import pytest

from application.budget import billing_principal
from infrastructure.budget.job_context import job_operations, next_operation_id
from infrastructure.metrics import metrics
from infrastructure.redis.redis_client import redis_client
from infrastructure.worker import runtime, sweep, tasks


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

    def create(self, *, master: bool = False) -> JobFactory:
        assert master
        return self

    async def __aenter__(self) -> SimpleNamespace:
        return SimpleNamespace(background_jobs=self.jobs)

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_ten_documents_from_same_principal_start_together() -> None:
    factory = JobFactory()
    started: set[int] = set()
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def process(command: ProcessDocumentCommand) -> None:
        started.add(command.document_id)
        assert billing_principal.get() == "user:9"
        assert job_operations.get() is not None

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
    assert billing_principal.get() is None
    assert job_operations.get() is None


@pytest.mark.asyncio
async def test_processing_retries_preserve_billing_scope() -> None:
    factory = JobFactory()
    observed = []

    @runtime.job()
    async def action(ctx: dict[str, int], **kwargs: object) -> None:
        observed.append((ctx["job_try"], billing_principal.get(), next_operation_id()))
        message = "failed"
        raise RuntimeError(message)

    ctx = {"uow_factory": factory}
    arguments = {"job_id": 5, "principal_id": 9, "cost_principal": "user:9", "cost_operation_id": "fixed"}

    for attempt in (1, 2):
        with pytest.raises(Retry) as raised:
            await action(dict(ctx, job_try=attempt), **arguments)
        assert raised.value.defer_score == 5000 * 2 ** (attempt - 1)

    await action(dict(ctx, job_try=3), **arguments)
    assert [row[0] for row in observed] == [1, 2, 3]
    assert all(row[1] == "user:9" for row in observed)
    assert len({row[2] for row in observed}) == 1
    factory.jobs.mark_failed.assert_awaited_once_with(5, "failed")
    factory.jobs.mark_done.assert_not_awaited()
    assert billing_principal.get() is None
    assert job_operations.get() is None


@pytest.mark.asyncio
async def test_cancellation_releases_billing_context() -> None:
    factory = JobFactory()

    @runtime.job()
    async def action(ctx: dict[str, int], **kwargs: object) -> None:
        raise asyncio.CancelledError

    ctx = {"uow_factory": factory}
    with pytest.raises(asyncio.CancelledError):
        await action(ctx, job_id=7, owner_id=8, cost_principal="user:8")
    factory.jobs.mark_failed.assert_awaited_once()
    assert billing_principal.get() is None
    assert job_operations.get() is None


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
