"""Shared worker policy preserves billing and processing retry boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from arq import Retry
import fakeredis.aioredis
import pytest

from application.budget import billing_principal
from config import settings
from domain.runtime.enums import LLMProvider
from infrastructure.budget.job_context import job_operations, next_operation_id
from infrastructure.metrics import metrics
from infrastructure.redis.redis_client import redis_client
from infrastructure.worker import runtime, sweep


class JobFactory:
    def __init__(self) -> None:
        self.jobs = SimpleNamespace(
            mark_running=AsyncMock(),
            mark_done=AsyncMock(),
            mark_failed=AsyncMock(),
            touch_heartbeat=AsyncMock(),
        )

    def create(self, *, master: bool = False) -> "JobFactory":
        assert master
        return self

    async def __aenter__(self) -> SimpleNamespace:
        return SimpleNamespace(background_jobs=self.jobs)

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_admission_waits_preserve_failure_retries_and_billing_scope() -> None:
    factory = JobFactory()
    observed = []

    @runtime.job(per_principal=True)
    async def action(ctx: dict[str, int], **kwargs: object) -> None:
        observed.append((ctx["job_try"], billing_principal.get(), next_operation_id()))
        message = "failed"
        raise RuntimeError(message)

    redis = fakeredis.aioredis.FakeRedis()
    ctx = {
        "redis": redis,
        "container": SimpleNamespace(infrastructure=SimpleNamespace(db=SimpleNamespace(uow_factory=factory))),
    }
    arguments = {"job_id": 5, "principal_id": 9, "cost_principal": "user:9", "cost_operation_id": "fixed"}
    try:
        await redis.set("rag:worker:principal:9", "other")

        for attempt in (1, 2):
            with pytest.raises(Retry):
                await action(dict(ctx, job_try=attempt), **arguments)

        factory.jobs.mark_running.assert_not_awaited()
        await redis.delete("rag:worker:principal:9")

        for attempt in (3, 4):
            with pytest.raises(Retry) as raised:
                await action(dict(ctx, job_try=attempt), **arguments)
            assert raised.value.defer_score == 5000 * 2 ** (attempt - 3)
            assert await redis.get("rag:worker:principal:9") is None

        await action(dict(ctx, job_try=5), **arguments)
        assert [row[0] for row in observed] == [1, 2, 3]
        assert all(row[1] == "user:9" for row in observed)
        assert len({row[2] for row in observed}) == 1
        factory.jobs.mark_failed.assert_awaited_once_with(5, "failed")
        factory.jobs.mark_done.assert_not_awaited()
        assert billing_principal.get() is None
        assert job_operations.get() is None
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_cancellation_releases_admission_and_billing_context() -> None:
    factory = JobFactory()

    @runtime.job(per_principal=True)
    async def action(ctx: dict[str, int], **kwargs: object) -> None:
        raise asyncio.CancelledError

    redis = fakeredis.aioredis.FakeRedis()
    ctx = {
        "redis": redis,
        "container": SimpleNamespace(infrastructure=SimpleNamespace(db=SimpleNamespace(uow_factory=factory))),
    }
    try:
        with pytest.raises(asyncio.CancelledError):
            await action(ctx, job_id=7, owner_id=8, cost_principal="user:8")
        factory.jobs.mark_failed.assert_awaited_once()
        assert await redis.get("rag:worker:principal:8") is None
        assert billing_principal.get() is None
        assert job_operations.get() is None
    finally:
        await redis.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [LLMProvider.OPENROUTER, LLMProvider.OLLAMA])
async def test_metrics_only_collect_ollama_for_selected_provider(
    monkeypatch: pytest.MonkeyPatch, provider: LLMProvider
) -> None:
    db, qdrant, ollama = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(metrics, "_collect_db_pool_metrics", db)
    monkeypatch.setattr(metrics, "_collect_qdrant_metrics", qdrant)
    monkeypatch.setattr(metrics, "_collect_ollama_metrics", ollama)
    monkeypatch.setattr(settings, "llm_provider", provider.value)
    await metrics.collect_infra_metrics()
    db.assert_awaited_once()
    qdrant.assert_awaited_once()
    assert ollama.await_count == int(provider == LLMProvider.OLLAMA)


@pytest.mark.asyncio
async def test_sweep_events_reuse_process_owned_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = SimpleNamespace(publish=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr(redis_client, "_redis", redis)
    await sweep.publish_sweep_event(4, {"evaluated": 1})
    await sweep.publish_sweep_event(4, {"done": True})
    assert redis.publish.await_count == 2
    assert redis.publish.await_args.args == ("sweep:4", '{"done": true}')
    redis.aclose.assert_not_awaited()
