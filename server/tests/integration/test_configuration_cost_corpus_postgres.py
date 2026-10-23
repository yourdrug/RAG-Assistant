"""Real PostgreSQL locks/journal with Redis Lua and a local Qdrant test double."""

import asyncio
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json
import os
import subprocess
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
import threading
from unittest.mock import AsyncMock

import httpx

import fakeredis.aioredis
import pytest
import pytest_asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from config import settings
from domain.shared.exceptions import ValidationError
from domain.runtime.cost import BudgetError, CostRate, ModelBillingContract
from application.budget import BudgetPolicy
from infrastructure.budget.job_context import bill_job_to_initiator
from infrastructure.budget.runtime import BudgetRuntime
from infrastructure.budget.transport import PaidAsyncTransport, PaidSyncTransport
from infrastructure.budget.ledger import PostgresRedisBudget
from infrastructure.corpus import control
from infrastructure.corpus.control import PostgresCorpusControl
from infrastructure.corpus.locks import corpus_write_guard
from infrastructure.database.database import DatabaseManager, DatabaseNode
from infrastructure.database.enums import DatabaseNodeRole
from infrastructure.database.dataclasses import DatabaseURL
from sqlalchemy.ext.asyncio import create_async_engine
from infrastructure.corpus.validation import corpus_digest
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.config.embedding_identity import configured_embedding_identity

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def configured_engine(pg_engine):
    return pg_engine


async def test_shared_money_admission_retry_reconciliation_and_redis_restart(configured_engine):
    sessions = async_sessionmaker(configured_engine, expire_on_commit=False)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    first = PostgresRedisBudget(sessions, redis)
    second = PostgresRedisBudget(sessions, redis)
    await first.add_rate(CostRate("openrouter", "m", 1, "tokens", Decimal("1"), Decimal("2"), 100))
    args = {
        "principal": "user:1",
        "provider": "openrouter",
        "model": "m",
        "operation": "chat",
        "maximum_units": 100,
        "rate_version": 1,
        "hourly_units": 500,
        "daily_units": 1000,
    }
    try:
        results = await asyncio.gather(
            *[(first if i % 2 else second).reserve(operation_id=f"op{i}", **args) for i in range(10)],
            return_exceptions=True,
        )
        assert results.count(True) == 5
        assert sum(isinstance(r, BudgetError) for r in results) == 5
        admitted = next(f"op{i}" for i, result in enumerate(results) if result is True)
        assert await second.reserve(operation_id=admitted, **args) is False
        await redis.flushall()
        with pytest.raises(BudgetError):
            await first.reserve(operation_id="after-restart", **args)
        await first.reconcile_pending(admitted, 0, "provider confirmed zero charge")
        await second.reconcile_pending(admitted, 0, "same confirmation")
        assert await second.reserve(operation_id="after-reconciliation", **args)
        await redis.aclose()
        with pytest.raises(ConnectionError):
            # An injected failure must roll the PostgreSQL reservation back.
            class OfflineRedis:
                async def eval(self, *args):
                    raise ConnectionError("offline")

            await PostgresRedisBudget(sessions, OfflineRedis()).reserve(operation_id="offline", **args)
        async with sessions() as session:
            assert (await session.execute(text("SELECT count(*) FROM cost_usage"))).scalar_one() == 6
            assert (
                await session.execute(
                    text("SELECT reconciliation_evidence FROM cost_usage WHERE operation_id=:id"),
                    {"id": admitted},
                )
            ).scalar_one() == "provider confirmed zero charge"
    finally:
        await redis.aclose()


@pytest_asyncio.fixture
async def corpus_pair(configured_engine, monkeypatch):
    sessions = async_sessionmaker(configured_engine, expire_on_commit=False)
    async with sessions() as session:
        schema = (await session.execute(text("SELECT current_schema()"))).scalar_one()
    url = configured_engine.url
    manager = DatabaseManager()
    # One ordinary pooled connection must remain available while a session lock is held.
    pooled = create_async_engine(
        url, pool_size=1, max_overflow=0, connect_args={"server_settings": {"search_path": schema}}
    )
    sessions = async_sessionmaker(pooled, expire_on_commit=False)
    manager.master_node = DatabaseNode(DatabaseNodeRole.MASTER, DatabaseURL("", "", "", "", ""), 0)
    manager.master_node.async_engine = pooled
    manager.master_node.async_session_factory = sessions
    manager.create_master_connection_engines(
        url.set(drivername="postgresql+psycopg"),
        {"options": f"-c search_path={schema}", "connect_timeout": 5},
    )
    monkeypatch.setattr(settings, "qdrant_url", "http://qdrant:6333")
    monkeypatch.setattr(settings, "collection_name", "v1")
    monkeypatch.setattr(settings, "embed_dim", 2)

    def qdrant(endpoint):
        client = QdrantClient(":memory:")
        fingerprint = hashlib.sha256(configured_embedding_identity().encode()).hexdigest()
        for name in ["v1", "v2"]:
            client.create_collection(
                name,
                vectors_config=VectorParams(size=2, distance=Distance.COSINE),
                sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
            )
            client.create_collection(
                name + "__embedding_identity",
                vectors_config={fingerprint: VectorParams(size=1, distance=Distance.COSINE)},
            )
        return client

    monkeypatch.setattr(control, "raw_corpus_client", qdrant)
    async with sessions() as session, session.begin():
        for key, value in [("qdrant_url", settings.qdrant_url), ("collection_name", "v1")]:
            await session.execute(
                text("""INSERT INTO config_parameters
                (key, value, value_type, category, creation_date)
                VALUES (:key, :value, 'str', 'corpus', now())"""),
                {"key": key, "value": value},
            )
    api_registry, worker_registry = MLClientRegistry(), MLClientRegistry()
    api = PostgresCorpusControl(manager, api_registry.corpus_resources, None)
    worker = PostgresCorpusControl(manager, worker_registry.corpus_resources, None)
    api_registry.corpus_resources.bind_readiness(api.is_ready)
    worker_registry.corpus_resources.bind_readiness(worker.is_ready)
    await api.start("api")
    await worker.start("worker")
    try:
        yield api, worker, sessions
    finally:
        await api.close()
        await worker.close()
        await api_registry.close()
        await worker_registry.close()
        await manager.disconnect()


async def test_corpus_switch_polls_without_notify_and_fences_stale_writers(corpus_pair):
    api, worker, sessions = corpus_pair
    old = api.registry.corpus_snapshot
    async with api.registry.corpus_operation():
        result = await api.switch(old.endpoint, "v2", old.generation)
        assert result["phase"] == "active"
        assert api.registry.qdrant_client().snapshot == old
        assert worker.registry.corpus_snapshot.physical_collection == "v2"
    new = api.registry.corpus_snapshot
    with pytest.raises(ValidationError):
        await asyncio.to_thread(lambda: run_write_guard(api.database, old))
    await asyncio.to_thread(lambda: run_write_guard(api.database, new))
    async with sessions() as session:
        assert (
            await session.execute(text("SELECT value FROM config_parameters WHERE key='collection_name'"))
        ).scalar_one() == "v2"


def run_write_guard(database, snapshot):
    with corpus_write_guard(database, snapshot):
        pass


async def test_concurrent_ingestion_aborts_before_commit_without_losing_db_write(corpus_pair, monkeypatch):
    api, worker, sessions = corpus_pair
    old = api.registry.corpus_snapshot
    wait = api.wait_prepared

    async def start_ingestion_after_probe(generation, required):
        await wait(generation, required)
        async with sessions() as session, session.begin():
            await session.execute(
                text("""INSERT INTO documents
                (filename, source_path, visibility, status, doc_domain, creation_date)
                VALUES ('new.txt', '', 'internal_public', 'pending', 'general', now())""")
            )

    monkeypatch.setattr(api, "wait_prepared", start_ingestion_after_probe)
    with pytest.raises(ValidationError, match="ingestion"):
        await api.switch(old.endpoint, "v2", old.generation)
    state = await api.read_state()
    assert state["phase"] == "failed"
    assert state["active"]["generation"] == old.generation
    assert api.registry.corpus_snapshot == old
    await asyncio.to_thread(lambda: run_write_guard(api.database, old))
    async with sessions() as session:
        assert (await session.execute(text("SELECT count(*) FROM documents"))).scalar_one() == 1


async def test_missing_worker_rejects_switch_and_committed_journal_recovers(corpus_pair):
    api, worker, sessions = corpus_pair
    old = api.registry.corpus_snapshot
    await worker.close()
    with pytest.raises(ValidationError, match="API and worker"):
        await api.switch(old.endpoint, "v2", old.generation)
    candidate = replace(old, alias="v2", physical_collection="v2", generation="recovered-generation")
    async with sessions() as session, session.begin():
        await session.execute(
            text("""UPDATE corpus_control SET phase='committing',
            active=CAST(:active AS jsonb), candidate=CAST(:active AS jsonb), digest=:digest WHERE id=1"""),
            {"active": json.dumps(asdict(candidate)), "digest": corpus_digest([])},
        )
        await session.execute(text("UPDATE config_parameters SET value='v2' WHERE key='collection_name'"))
    with pytest.raises(ValidationError, match="pending"):
        await asyncio.to_thread(lambda: run_write_guard(api.database, candidate))
    await api.tick()
    await worker.start("worker")
    await api.tick()
    assert (await api.read_state())["phase"] == "active"
    assert api.registry.corpus_snapshot == candidate
    assert worker.registry.corpus_snapshot == candidate
    await asyncio.to_thread(lambda: run_write_guard(api.database, candidate))


async def test_switch_waits_for_admitted_writer_before_publishing(corpus_pair):
    api, worker, sessions = corpus_pair
    old = api.registry.corpus_snapshot
    started, release = threading.Event(), threading.Event()

    def admitted_write():
        with corpus_write_guard(api.database, old):
            started.set()
            if not release.wait(10):
                raise TimeoutError("test did not release writer")

    writing = asyncio.create_task(asyncio.to_thread(admitted_write))
    switching = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        switching = asyncio.create_task(api.switch(old.endpoint, "v2", old.generation))
        await asyncio.sleep(0.1)
        assert not switching.done()
        assert api.registry.corpus_snapshot == old
        release.set()
        await writing
        with pytest.raises(ValidationError, match="changed during validation"):
            await switching
        assert (await api.switch(old.endpoint, "v2", old.generation))["phase"] == "active"
    finally:
        release.set()
        await asyncio.gather(writing, *([switching] if switching is not None else []), return_exceptions=True)


async def test_budget_policy_reads_committed_database_without_notification(configured_engine, monkeypatch):
    from infrastructure.budget.runtime import BudgetRuntime

    sessions = async_sessionmaker(configured_engine, expire_on_commit=False)
    monkeypatch.setattr(settings, "cost_rate_limit_enabled", False)
    async with sessions() as session, session.begin():
        for key, value in {
            "cost_rate_limit_enabled": "true",
            "cost_hourly_limit": "0.00012",
            "cost_daily_limit": "1",
        }.items():
            await session.execute(
                text("""INSERT INTO config_parameters
                (key, value, domain_key, value_type, category, creation_date)
                VALUES (:key, :value, NULL, 'str', 'cost', now())"""),
                {"key": key, "value": value},
            )
    runtime = BudgetRuntime()
    runtime.session_factory = sessions
    runtime.bootstrapped = True
    policy = await runtime.policy_snapshot()
    assert policy.enabled is True
    assert policy.hourly_usd == "0.00012"
    async with sessions() as session, session.begin():
        await session.execute(
            text("UPDATE config_parameters SET value='0.00006' WHERE key='cost_hourly_limit'")
        )
    assert (await runtime.policy_snapshot()).hourly_usd == "0.00006"
    async with sessions() as session, session.begin():
        await session.execute(text("DELETE FROM config_parameters WHERE key='cost_daily_limit'"))
    with pytest.raises(BudgetError, match="not initialized"):
        await runtime.policy_snapshot()


async def test_physical_sync_async_real_redis_restart_and_tariff_debt(configured_engine):
    redis_url = os.environ.get("TEST_REDIS_URL")
    container = os.environ.get("TEST_REDIS_CONTAINER")
    if not redis_url or not container:
        pytest.skip("Requires isolated TEST_REDIS_URL and TEST_REDIS_CONTAINER for a real restart")
    sessions = async_sessionmaker(configured_engine, expire_on_commit=False)
    redis = Redis.from_url(redis_url, decode_responses=True)
    ledger = PostgresRedisBudget(sessions, redis)
    await ledger.add_rate(CostRate("openrouter", "m", 1, "tokens", Decimal("1"), Decimal("2"), 100))
    principal = "user:" + uuid4().hex
    args = {
        "principal": principal,
        "provider": "openrouter",
        "model": "m",
        "operation": "llm",
        "maximum_units": 100,
        "rate_version": 1,
        "hourly_units": 500,
        "daily_units": 500,
    }
    try:
        # Sync callers and async callers enter exactly the same PostgreSQL lock and Lua gate.
        loop = asyncio.get_running_loop()

        def reserve_sync(index):
            return asyncio.run_coroutine_threadsafe(
                ledger.reserve(operation_id=f"sync{index}", **args),
                loop,
            ).result()

        results = await asyncio.gather(
            *(
                asyncio.to_thread(reserve_sync, i)
                if i % 2
                else ledger.reserve(operation_id=f"async{i}", **args)
                for i in range(10)
            ),
            return_exceptions=True,
        )
        assert results.count(True) == 5
        assert sum(isinstance(result, BudgetError) for result in results) == 5
        await asyncio.to_thread(
            subprocess.run, ["docker", "restart", container], check=True, capture_output=True
        )
        await redis.aclose()
        redis = Redis.from_url(redis_url, decode_responses=True)
        ledger.redis = redis
        async with asyncio.timeout(5):
            while True:
                try:
                    await redis.ping()
                    break
                except RedisConnectionError:
                    await asyncio.sleep(0.05)
        with pytest.raises(BudgetError, match="budget exceeded"):
            await ledger.reserve(operation_id="after-real-restart", **args)
        admitted = next(
            f"sync{i}" if i % 2 else f"async{i}" for i, result in enumerate(results) if result is True
        )
        await ledger.reconcile(
            principal, admitted, 0, billable_units={"input_tokens": 0}, cost_source="provider"
        )
        await ledger.reconcile(
            principal, admitted, 0, billable_units={"input_tokens": 0}, cost_source="provider"
        )
        with pytest.raises(BudgetError, match="Conflicting"):
            await ledger.reconcile(principal, admitted, 1)
        assert await ledger.reserve(operation_id="disabled-quota", **dict(args, enforce_limits=False))
        with pytest.raises(BudgetError, match="exceeded reservation"):
            await ledger.reconcile(principal, "disabled-quota", 101)
        with pytest.raises(BudgetError, match="tariff"):
            await ledger.reserve(operation_id="violated-tariff", **dict(args, enforce_limits=False))
    finally:
        await redis.aclose()


async def test_audit_preserves_pending_and_settled_usage(configured_engine):
    async with configured_engine.begin() as connection:
        await connection.execute(
            text("""
            INSERT INTO cost_rates(provider, model, version, unit, input_usd_per_million,
                                   output_usd_per_million, context_tokens)
            VALUES ('openrouter', 'legacy', 3, 'tokens', 1, 2, 100)
        """)
        )
        await connection.execute(
            text("""
            INSERT INTO cost_usage(operation_id, principal, provider, model, operation,
                                   rate_version, reserved_units, actual_units, settled_at, dispatch_state)
            VALUES ('pending', 'user:1', 'openrouter', 'legacy', 'llm', 3, 100, NULL, NULL, 'sending'),
                   ('settled', 'user:1', 'openrouter', 'legacy', 'llm', 3, 100, 70, now(), 'settled')
        """)
        )
    sessions = async_sessionmaker(configured_engine, expire_on_commit=False)
    ledger = PostgresRedisBudget(sessions, None)
    rows = {row["operation_id"]: row for row in await ledger.usage()}
    assert rows["pending"]["dispatch_state"] == "sending"
    assert rows["settled"]["dispatch_state"] == "settled"
    assert rows["settled"]["actual_units"] == 70
    assert (await ledger.rates())[0]["version"] == 3
    assert len(await ledger.pending()) == 1


@pytest.mark.parametrize("sync", [False, True])
async def test_budget_worker_retries_only_provably_undispatched_attempt(configured_engine, monkeypatch, sync):
    sessions = async_sessionmaker(configured_engine, expire_on_commit=False)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ledger = PostgresRedisBudget(sessions, redis)
    rate = CostRate("openrouter", "m", 1, "tokens", Decimal("1"), Decimal("2"), 100)
    await ledger.add_rate(rate)
    runtime = BudgetRuntime()
    runtime.configure(ledger, sessions, AsyncMock())
    runtime.catalog.contract.return_value = ModelBillingContract(
        "tokens", "llm", 100, frozenset({"model", "messages", "max_tokens"}), "test"
    )
    runtime.rate = AsyncMock(return_value=rate)
    runtime.policy_snapshot = AsyncMock(return_value=BudgetPolicy(True))
    mark_dispatch = ledger.mark_dispatch
    calls = 0
    sent = []

    async def fail_once_before_send(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Simulate cancellation/error after the dispatch marker commits but before HTTP send.
            await mark_dispatch(*args)
            raise ConnectionError("Before provider dispatch")
        await mark_dispatch(*args)

    monkeypatch.setattr(ledger, "mark_dispatch", fail_once_before_send)

    def provider(request):
        sent.append(True)
        return httpx.Response(200, json={"id": "gen-success", "usage": {"cost": "0"}})

    def invoke_sync():
        with httpx.Client(
            transport=PaidSyncTransport(runtime=runtime, transport=httpx.MockTransport(provider))
        ) as client:
            return client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"model": "m", "messages": [], "max_tokens": 10},
            )

    @bill_job_to_initiator
    async def job(ctx):
        if sync:
            return await asyncio.to_thread(invoke_sync)
        async with httpx.AsyncClient(
            transport=PaidAsyncTransport(runtime=runtime, transport=httpx.MockTransport(provider))
        ) as client:
            return await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"model": "m", "messages": [], "max_tokens": 10},
            )

    try:
        with pytest.raises(ConnectionError, match="Before provider"):
            await job({"job_id": "job"}, cost_principal="user:7")
        assert not sent
        rows = await ledger.usage()
        assert len(rows) == 1
        assert rows[0]["actual_units"] == 0
        response = await job({"job_id": "job"}, cost_principal="user:7")
        assert response.status_code == 200
        assert sent == [True]
        rows = await ledger.usage()
        assert len(rows) == 1
        assert rows[0]["provider_response_id"] == "gen-success"
        assert rows[0]["cost_source"] == "provider"
        # A real provider response costing zero still must never be replayed.
        with pytest.raises(BudgetError, match="already reserved"):
            await job({"job_id": "job"}, cost_principal="user:7")
        assert sent == [True]
    finally:
        await redis.aclose()


async def test_budget_undispatched_retry_is_atomic_and_rechecks_quota(configured_engine):
    sessions = async_sessionmaker(configured_engine, expire_on_commit=False)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ledger = PostgresRedisBudget(sessions, redis)
    await ledger.add_rate(CostRate("openrouter", "m", 1, "tokens", Decimal("1"), Decimal("2"), 100))
    args = {
        "principal": "user:1",
        "provider": "openrouter",
        "model": "m",
        "operation": "llm",
        "maximum_units": 100,
        "rate_version": 1,
        "hourly_units": 100,
        "daily_units": 100,
        "request_fingerprint": "same-request",
    }
    try:
        assert await ledger.reserve(operation_id="retry", **args)
        await ledger.reconcile("user:1", "retry", 0, cost_source="not_dispatched")
        assert await ledger.reserve(operation_id="other", **args)
        with pytest.raises(BudgetError, match="budget exceeded"):
            await ledger.reserve(operation_id="retry", **args)
        await ledger.reconcile("user:1", "other", 0)
        assert (
            await ledger.reserve(operation_id="retry", **dict(args, request_fingerprint="different-request"))
            is False
        )
        results = await asyncio.gather(*(ledger.reserve(operation_id="retry", **args) for _ in range(8)))
        assert results.count(True) == 1
        assert results.count(False) == 7
        pending = await ledger.pending()
        assert len(pending) == 1
        assert pending[0]["reserved_units"] == 100
        assert pending[0]["actual_units"] is None
        assert pending[0]["dispatch_state"] == "reserved"
        with pytest.raises(BudgetError, match="another principal"):
            await ledger.reserve(operation_id="retry", **dict(args, principal="user:2"))
    finally:
        await redis.aclose()


@pytest.mark.parametrize("source", ["provider", "administrative", None])
async def test_budget_zero_charge_or_pending_is_not_permission_to_replay(configured_engine, source):
    sessions = async_sessionmaker(configured_engine, expire_on_commit=False)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    ledger = PostgresRedisBudget(sessions, redis)
    await ledger.add_rate(CostRate("openrouter", "m", 1, "tokens", Decimal("1"), Decimal("2"), 100))
    args = {
        "principal": "user:1",
        "provider": "openrouter",
        "model": "m",
        "operation": "llm",
        "maximum_units": 100,
        "rate_version": 1,
        "hourly_units": 100,
        "daily_units": 100,
        "request_fingerprint": "same-request",
    }
    try:
        assert await ledger.reserve(operation_id="op", **args)
        await ledger.mark_dispatch("user:1", "op")
        if source is not None:
            await ledger.reconcile("user:1", "op", 0, cost_source=source)
        assert await ledger.reserve(operation_id="op", **args) is False
    finally:
        await redis.aclose()


@pytest.mark.parametrize("failure", [False, True])
async def test_write_guard_uses_autocommit_and_releases_session_lock(corpus_pair, failure):
    api, worker, sessions = corpus_pair
    snapshot = api.registry.corpus_snapshot
    entered, release = threading.Event(), threading.Event()

    def write():
        with corpus_write_guard(api.database, snapshot):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release writer")
            if failure:
                raise RuntimeError("Qdrant write failed")

    writing = asyncio.create_task(asyncio.to_thread(write))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        async with sessions() as session:
            # The lock owner must be idle, with no transaction during the external write.
            owners = (
                await session.execute(
                    text("""SELECT a.state, a.xact_start
                FROM pg_locks l JOIN pg_stat_activity a ON a.pid=l.pid
                WHERE l.locktype='advisory' AND l.granted
                  AND a.datname=current_database() AND a.pid<>pg_backend_pid()""")
                )
            ).all()
            assert owners
            assert all(state == "idle" and transaction is None for state, transaction in owners)
        release.set()
        if failure:
            with pytest.raises(RuntimeError, match="Qdrant"):
                await writing
        else:
            await writing
        await assert_corpus_lock_available(api.database)
        assert (await api.read_state())["write_epoch"] == 1
    finally:
        release.set()
        await asyncio.gather(writing, return_exceptions=True)


async def assert_corpus_lock_available(database):
    from infrastructure.corpus.locks import CORPUS_SWITCH_ADVISORY_LOCK_KEY

    async with asyncio.timeout(2):
        while True:
            async with database.async_master_connection() as connection:
                available = (
                    await connection.execute(
                        text("SELECT pg_try_advisory_lock(:key)"),
                        {"key": CORPUS_SWITCH_ADVISORY_LOCK_KEY},
                    )
                ).scalar_one()
                if available:
                    return
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("failure", ["cancel", "timeout", "error"])
async def test_switch_releases_lock_and_recovers_preparation_on_interruption(
    corpus_pair, monkeypatch, failure
):
    api, worker, sessions = corpus_pair
    snapshot = api.registry.corpus_snapshot
    entered = asyncio.Event()

    async def interrupted_prepare(*args):
        entered.set()
        if failure == "error":
            raise RuntimeError("preparation failed")
        if failure == "timeout":
            async with asyncio.timeout(0.02):
                await asyncio.Event().wait()
        await asyncio.Event().wait()

    monkeypatch.setattr(api, "wait_prepared", interrupted_prepare)
    switching = asyncio.create_task(api.switch(snapshot.endpoint, "v2", snapshot.generation))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if failure == "cancel":
            switching.cancel()
        expected = {"cancel": asyncio.CancelledError, "timeout": TimeoutError, "error": RuntimeError}[failure]
        with pytest.raises(expected):
            await switching
        state = await api.read_state()
        assert state["phase"] == "failed"
        assert state["active"]["generation"] == snapshot.generation
        await assert_corpus_lock_available(api.database)
        await asyncio.to_thread(run_write_guard, api.database, snapshot)
    finally:
        if not switching.done():
            switching.cancel()
        await asyncio.gather(switching, return_exceptions=True)


@pytest.mark.parametrize("cancel", [False, True])
async def test_lock_acquisition_timeout_or_cancel_closes_dedicated_connection(corpus_pair, cancel):
    from infrastructure.corpus.locks import corpus_switch_guard
    from sqlalchemy.exc import DBAPIError

    api, worker, sessions = corpus_pair
    entered = asyncio.Event()

    async def wait_for_lock():
        entered.set()
        async with corpus_switch_guard(api.database):
            pytest.fail("competing exclusive lock was admitted")

    async with corpus_switch_guard(api.database):
        waiting = asyncio.create_task(wait_for_lock())
        try:
            await entered.wait()
            await asyncio.sleep(0.05)
            if cancel:
                waiting.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else DBAPIError):
                await asyncio.wait_for(waiting, 7)
        finally:
            if not waiting.done():
                waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
    await assert_corpus_lock_available(api.database)


async def test_concurrent_switches_publish_only_one_generation(corpus_pair):
    api, worker, sessions = corpus_pair
    snapshot = api.registry.corpus_snapshot
    results = await asyncio.gather(
        api.switch(snapshot.endpoint, 'v2', snapshot.generation),
        worker.switch(snapshot.endpoint, 'v2', snapshot.generation),
        return_exceptions=True,
    )
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, ValidationError) for result in results) == 1
    state = await api.read_state()
    assert state['phase'] == 'active'
    assert state['active']['generation'] != snapshot.generation
    await assert_corpus_lock_available(api.database)


async def test_orphaned_preparation_recovers_without_publishing_candidate(corpus_pair):
    api, worker, sessions = corpus_pair
    snapshot = api.registry.corpus_snapshot
    candidate = replace(snapshot, alias='v2', physical_collection='v2', generation='orphan')
    await api.repository.begin_prepare(candidate, corpus_digest([]), 'dead-process')
    await api.tick()
    state = await api.read_state()
    assert state['phase'] == 'failed'
    assert state['active']['generation'] == snapshot.generation
    assert state['candidate'] is None
    await asyncio.to_thread(run_write_guard, api.database, snapshot)
