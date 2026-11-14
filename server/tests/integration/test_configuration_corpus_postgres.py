"""Real PostgreSQL locks/journal with Redis Lua and a local Qdrant test double."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import threading
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from config import settings
from domain.shared.exceptions import ValidationError
from infrastructure.corpus import control
from infrastructure.corpus.control import PostgresCorpusControl
from infrastructure.corpus.locks import (
    CORPUS_SWITCH_ADVISORY_LOCK_KEY,
    corpus_switch_guard,
    corpus_write_guard,
)
from infrastructure.corpus.validation import corpus_digest
from infrastructure.database.database import DatabaseManager, DatabaseNode
from infrastructure.database.dataclasses import DatabaseURL
from infrastructure.database.enums import DatabaseNodeRole
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.ml.config.embedding_identity import configured_embedding_identity
from infrastructure.repositories.vector.sparse_vectors import SPARSE_VECTOR_NAME


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from infrastructure.corpus.snapshot import CorpusSnapshot


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def configured_engine(pg_engine: AsyncEngine) -> AsyncEngine:
    return pg_engine


@pytest_asyncio.fixture
async def corpus_pair(  # noqa: PLR0915 - One fixture assembles both corpus participants.
    configured_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[PostgresCorpusControl, PostgresCorpusControl, async_sessionmaker[AsyncSession]]]:
    sessions = async_sessionmaker(configured_engine, expire_on_commit=False)
    async with sessions() as session:
        schema = (await session.execute(text("SELECT current_schema()"))).scalar_one()
    url = configured_engine.url
    manager = DatabaseManager(settings)
    # One ordinary pooled connection must remain available while a session lock is held.
    pooled = create_async_engine(
        url, pool_size=1, max_overflow=0, connect_args={"server_settings": {"search_path": schema}}
    )
    sessions = async_sessionmaker(pooled, expire_on_commit=False)
    manager.master_node = DatabaseNode(DatabaseNodeRole.MASTER, DatabaseURL("", "", "", "", ""), 0, settings)
    manager.master_node.async_engine = pooled
    manager.master_node.async_session_factory = sessions
    manager.create_master_connection_engines(
        url.set(drivername="postgresql+psycopg"),
        {"options": f"-c search_path={schema}", "connect_timeout": 5},
    )
    monkeypatch.setattr(settings, "qdrant_url", "http://qdrant:6333")
    monkeypatch.setattr(settings, "collection_name", "v1")
    monkeypatch.setattr(settings, "embed_dim", 2)

    def qdrant(endpoint: object) -> QdrantClient:
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


async def test_corpus_switch_polls_without_notify_and_fences_stale_writers(corpus_pair: Any) -> None:
    api, worker, sessions = corpus_pair
    old = api.resources.corpus_snapshot
    async with api.resources.corpus_operation():
        result = await api.switch(old.endpoint, "v2", old.generation)
        assert result["phase"] == "active"
        assert api.resources.qdrant_client().snapshot == old
        assert worker.resources.corpus_snapshot.physical_collection == "v2"
    new = api.resources.corpus_snapshot
    with pytest.raises(ValidationError):
        await asyncio.to_thread(lambda: run_write_guard(api.database, old))
    await asyncio.to_thread(lambda: run_write_guard(api.database, new))
    async with sessions() as session:
        assert (
            await session.execute(text("SELECT value FROM config_parameters WHERE key='collection_name'"))
        ).scalar_one() == "v2"


def run_write_guard(database: Any, snapshot: CorpusSnapshot) -> None:
    with corpus_write_guard(database, snapshot):
        pass


async def test_concurrent_ingestion_aborts_before_commit_without_losing_db_write(
    corpus_pair: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    api, _worker, sessions = corpus_pair
    old = api.resources.corpus_snapshot
    wait = api.wait_prepared

    async def start_ingestion_after_probe(generation: Any, required: Any) -> None:
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
    assert api.resources.corpus_snapshot == old
    await asyncio.to_thread(lambda: run_write_guard(api.database, old))
    async with sessions() as session:
        assert (await session.execute(text("SELECT count(*) FROM documents"))).scalar_one() == 1


async def test_missing_worker_rejects_switch_and_committed_journal_recovers(corpus_pair: Any) -> None:
    api, worker, sessions = corpus_pair
    old = api.resources.corpus_snapshot
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
    assert api.resources.corpus_snapshot == candidate
    assert worker.resources.corpus_snapshot == candidate
    await asyncio.to_thread(lambda: run_write_guard(api.database, candidate))


async def test_switch_waits_for_admitted_writer_before_publishing(corpus_pair: Any) -> None:
    api, _worker, _sessions = corpus_pair
    old = api.resources.corpus_snapshot
    started, release = threading.Event(), threading.Event()

    def admitted_write() -> None:
        with corpus_write_guard(api.database, old):
            started.set()

            if not release.wait(10):
                msg = "test did not release writer"
                raise TimeoutError(msg)

    writing = asyncio.create_task(asyncio.to_thread(admitted_write))
    switching = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        switching = asyncio.create_task(api.switch(old.endpoint, "v2", old.generation))
        await asyncio.sleep(0.1)
        assert not switching.done()
        assert api.resources.corpus_snapshot == old
        release.set()
        await writing
        with pytest.raises(ValidationError, match="changed during validation"):
            await switching
        assert (await api.switch(old.endpoint, "v2", old.generation))["phase"] == "active"
    finally:
        release.set()
        await asyncio.gather(writing, *([switching] if switching is not None else []), return_exceptions=True)


@pytest.mark.parametrize("failure", [False, True])
async def test_write_guard_uses_autocommit_and_releases_session_lock(corpus_pair: Any, failure: bool) -> None:  # noqa: PLR0915 - One scenario verifies the complete transaction/lifecycle.
    api, _worker, sessions = corpus_pair
    snapshot = api.resources.corpus_snapshot
    entered, release = threading.Event(), threading.Event()

    def write() -> None:
        with corpus_write_guard(api.database, snapshot):
            entered.set()

            if not release.wait(10):
                msg = "test did not release writer"
                raise TimeoutError(msg)

            if failure:
                msg = "Qdrant write failed"
                raise RuntimeError(msg)

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


async def assert_corpus_lock_available(database: Any) -> None:

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
async def test_switch_releases_lock_and_recovers_preparation_on_interruption(  # noqa: PLR0915 - One scenario verifies the complete transaction/lifecycle.
    corpus_pair: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    api, _worker, _sessions = corpus_pair
    snapshot = api.resources.corpus_snapshot
    entered = asyncio.Event()

    async def interrupted_prepare(*args: Any) -> None:
        entered.set()

        if failure == "error":
            msg = "preparation failed"
            raise RuntimeError(msg)

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
async def test_lock_acquisition_timeout_or_cancel_closes_dedicated_connection(
    corpus_pair: Any, cancel: bool
) -> None:

    api, _worker, _sessions = corpus_pair
    entered = asyncio.Event()

    async def wait_for_lock() -> None:
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


async def test_concurrent_switches_publish_only_one_generation(corpus_pair: Any) -> None:
    api, worker, _sessions = corpus_pair
    snapshot = api.resources.corpus_snapshot
    results = await asyncio.gather(
        api.switch(snapshot.endpoint, "v2", snapshot.generation),
        worker.switch(snapshot.endpoint, "v2", snapshot.generation),
        return_exceptions=True,
    )
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(isinstance(result, ValidationError) for result in results) == 1
    state = await api.read_state()
    assert state["phase"] == "active"
    assert state["active"]["generation"] != snapshot.generation
    await assert_corpus_lock_available(api.database)


async def test_orphaned_preparation_recovers_without_publishing_candidate(corpus_pair: Any) -> None:
    api, _worker, _sessions = corpus_pair
    snapshot = api.resources.corpus_snapshot
    candidate = replace(snapshot, alias="v2", physical_collection="v2", generation="orphan")
    await api.repository.begin_prepare(candidate, corpus_digest([]), "dead-process")
    await api.tick()
    state = await api.read_state()
    assert state["phase"] == "failed"
    assert state["active"]["generation"] == snapshot.generation
    assert state["candidate"] is None
    await asyncio.to_thread(run_write_guard, api.database, snapshot)
