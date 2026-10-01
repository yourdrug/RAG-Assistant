"""Regression contracts for medium lifecycle, retrieval and transaction findings."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from langchain.schema import Document as LCDocument

from application.services.retrieval import HybridRetriever
from domain.entities.document import Document
from infrastructure.ml import answer_cache
from infrastructure.ml.clients.managed_llm import ManagedLLM, ManagedInstructor
from infrastructure.redis.redis_client import RedisClient
from infrastructure.repositories.vector.embedding_identity import ensure_embedding_identity
from infrastructure.worker.admission import one_document_per_principal
from fakes import FakeUnitOfWorkFactory


@pytest.mark.asyncio
async def test_fake_factory_creates_transactions_and_rolls_back():
    factory = FakeUnitOfWorkFactory()
    async with factory.create(master=True) as first:
        saved = await first.documents.save(Document(filename="committed.pdf"))
    assert first._committed
    with pytest.raises(ValueError):
        async with factory.create(master=True) as second:
            await second.documents.delete(saved.id)
            raise ValueError("abort")
    assert first is not second
    assert second._rolled_back and not second._committed
    async with factory.create() as third:
        assert await third.documents.get_by_id(saved.id) is not None


@pytest.mark.asyncio
async def test_redis_failed_ping_closes_client_and_allows_retry(monkeypatch):
    failed = MagicMock(ping=AsyncMock(side_effect=OSError("offline")), aclose=AsyncMock())
    healthy = MagicMock(ping=AsyncMock(), aclose=AsyncMock())
    factory = MagicMock(side_effect=[failed, healthy])
    import redis.asyncio as aioredis

    monkeypatch.setattr(aioredis, "from_url", factory)
    client = RedisClient()
    with pytest.raises(OSError):
        await client.init()
    failed.aclose.assert_awaited_once()
    with pytest.raises(RuntimeError):
        _ = client.async_redis
    await client.init()
    assert client.async_redis is healthy
    await client.aclose()


@pytest.mark.asyncio
async def test_llm_retirement_waits_for_active_calls_and_closes_pool():
    started, finish = asyncio.Event(), asyncio.Event()
    pool = SimpleNamespace(close=AsyncMock())

    async def invoke(*args, **kwargs):
        started.set()
        await finish.wait()
        return "answer"

    raw = SimpleNamespace(ainvoke=invoke, root_async_client=pool, root_client=None)
    managed = ManagedLLM(raw)
    call = asyncio.create_task(managed.ainvoke("question"))
    await started.wait()
    managed.retire()
    await asyncio.sleep(0)
    pool.close.assert_not_awaited()
    finish.set()
    assert await call == "answer"
    await managed.close()
    pool.close.assert_awaited_once()
    with pytest.raises(RuntimeError, match="retired"):
        await managed.ainvoke("late")


@pytest.mark.asyncio
async def test_instructor_retirement_closes_underlying_openai():
    pool = SimpleNamespace(close=AsyncMock())
    create = AsyncMock(return_value="structured")
    raw = SimpleNamespace(client=pool, chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    managed = ManagedInstructor(raw)
    assert await managed.chat.completions.create(model="m") == "structured"
    await managed.close()
    pool.close.assert_awaited_once()


def test_full_rrf_ranks_sparse_first_and_limits_fetch_k():
    dense = LCDocument(page_content="dense text")
    sparse = LCDocument(page_content="sparse text")
    result = HybridRetriever().merge_and_dedup(
        [("dense", 0.9)],
        [("sparse", 2.0)],
        {"dense": (0.9, dense), "sparse": (0, sparse)},
        fetch_k=1,
        rrf_k=30,
        dense_weight=1,
        sparse_weight=3,
    )
    assert result == [sparse]


@pytest.mark.asyncio
async def test_cache_store_hit_isolation_invalidate_miss_without_scan(monkeypatch):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(answer_cache.redis_client, "_redis", redis)
    question = answer_cache.compute_question_hash("same question")
    user_a = answer_cache.compute_visibility_scope_hash("internal", 1, [], "user")
    user_b = answer_cache.compute_visibility_scope_hash("internal", 2, [], "user")
    admin_a = answer_cache.compute_visibility_scope_hash("internal", 1, [], "admin")
    try:
        await answer_cache.store_cached_answer("same question", question, "private answer", [], user_a, [42])
        assert (await answer_cache.find_cached_answer(question, user_a))["answer"] == "private answer"
        assert await answer_cache.find_cached_answer(question, user_b) is None
        assert await answer_cache.find_cached_answer(question, admin_a) is None
        assert await answer_cache.invalidate_by_document_ids([42]) == 1
        assert await answer_cache.find_cached_answer(question, user_a) is None
        assert not await redis.smembers(answer_cache._doc_index_key(42))
        monkeypatch.setattr(redis, "scan_iter", MagicMock(side_effect=AssertionError("full scan")))
        assert await answer_cache.invalidate_by_document_ids([999]) == 0
    finally:
        await redis.aclose()


def test_embedding_binding_rejects_same_dimension_different_model():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    try:
        assert ensure_embedding_identity(client, "provider:model-a:v1") == "provider:model-a:v1"
        with pytest.raises(ValueError, match="different embedding model"):
            ensure_embedding_identity(client, "provider:model-b:v1")
    finally:
        client.close()


@pytest.mark.asyncio
async def test_document_worker_defers_same_owner_and_admits_other_owner():
    from arq import Retry

    redis = fakeredis.aioredis.FakeRedis()
    started, finish = asyncio.Event(), asyncio.Event()
    ran = []

    @one_document_per_principal
    async def action(ctx, **kwargs):
        ran.append(kwargs["job_id"])
        if kwargs["job_id"] == 1:
            started.set()
            await finish.wait()

    first = asyncio.create_task(action({"redis": redis}, owner_id=10, principal_id=1, job_id=1))
    try:
        await started.wait()
        with pytest.raises(Retry):
            await action({"redis": redis}, owner_id=20, principal_id=1, job_id=2)
        await action({"redis": redis}, owner_id=10, principal_id=2, job_id=3)
        assert ran == [1, 3]
        finish.set()
        await first
        await action({"redis": redis, "job_try": 2}, owner_id=20, principal_id=1, job_id=2)
        assert ran == [1, 3, 2]
    finally:
        finish.set()
        await first
        await redis.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,index_method", [("edit", "replace"), ("add", "add"), ("delete", "remove")]
)
@pytest.mark.parametrize("fail_commit", [False, True])
async def test_bm25_mutates_only_after_successful_commit(monkeypatch, operation, index_method, fail_commit):
    from application.services.chunk_mutation_service import ChunkMutationService
    from domain.value_objects.document_status import DocumentStatus

    events = []
    document = Document(id=1, filename="test.pdf", status=DocumentStatus.DONE)
    chunk = SimpleNamespace(document_id=1, content_hash="old", chunk_index=0, manual=False)
    uow = SimpleNamespace(
        chunks=SimpleNamespace(
            get_by_id=AsyncMock(return_value=chunk),
            find_duplicate_by_hash=AsyncMock(return_value=None),
            update_content=AsyncMock(),
            get_max_chunk_index=AsyncMock(return_value=0),
            insert_one=AsyncMock(return_value=2),
            delete_one=AsyncMock(),
        ),
        documents=SimpleNamespace(set_has_manual_edits=AsyncMock()),
        vector_outbox=SimpleNamespace(enqueue=AsyncMock()),
    )

    class Transaction:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *args):
            if fail_commit:
                raise OSError("commit failed")
            events.append("committed")

    index = MagicMock()
    getattr(index, index_method).side_effect = lambda *args, **kwargs: events.append("indexed")
    factory = SimpleNamespace(create=lambda **kwargs: Transaction())
    service = ChunkMutationService(factory, SimpleNamespace(chunk_size=100), index)
    service._update_document_stats = AsyncMock()
    monkeypatch.setattr(
        "application.services.chunk_mutation_service.load_doc_for_edit",
        AsyncMock(return_value=(document, None)),
    )
    monkeypatch.setattr(
        "application.services.chunk_mutation_service.load_doc_for_add",
        AsyncMock(return_value=(document, None)),
    )
    if operation == "edit":
        action = service.edit_chunk(1, 2, "x" * 80, 1, "admin")
    elif operation == "add":
        action = service.add_chunk(1, "x" * 80, 1, "admin")
    else:
        action = service.delete_chunk(1, 2, 1, "admin")
    if fail_commit:
        with pytest.raises(OSError, match="commit failed"):
            await action
        assert events == []
        getattr(index, index_method).assert_not_called()
    else:
        await action
        assert events == ["committed", "indexed"]


@pytest.mark.asyncio
async def test_lifespan_cleans_up_when_database_startup_fails(monkeypatch):
    import main

    monkeypatch.setattr(main.logging.config, "dictConfig", MagicMock())
    monkeypatch.setattr(main, "attach_log_buffer", MagicMock())
    monkeypatch.setattr(main.redis_client, "init", AsyncMock())
    monkeypatch.setattr(main.database, "connect", AsyncMock(side_effect=OSError("database offline")))
    cleanup = AsyncMock()
    monkeypatch.setattr(main, "_shutdown", cleanup)
    application = SimpleNamespace(state=SimpleNamespace())
    with pytest.raises(OSError, match="database offline"):
        async with main.lifespan(application):
            pytest.fail("startup should fail before yield")
    cleanup.assert_awaited_once_with(None, main.scheduler, main.database, main.redis_client)
    assert application.state.container is None


@pytest.mark.asyncio
async def test_cache_shared_indexes_disabled_and_corrupt_entries(monkeypatch):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(answer_cache.redis_client, "_redis", redis)
    try:
        for question in ("q1", "q2"):
            await answer_cache.store_cached_answer(question, question, "answer", [], "scope", [1, 2])
        assert await answer_cache.invalidate_by_document_ids([1]) == 2
        assert not await redis.smembers(answer_cache._doc_index_key(2))
        await answer_cache.store_cached_answer("q", "q", "answer", [], "scope", cache_enabled=False)
        assert await answer_cache.find_cached_answer("q", "scope", cache_enabled=False) is None
        assert await answer_cache.invalidate_by_document_ids([], cache_enabled=True) == 0
        assert await answer_cache.invalidate_by_document_ids([1], cache_enabled=False) == 0
        key = answer_cache._cache_key("broken", "scope")
        await redis.set(key, "invalid base64")
        assert await answer_cache.find_cached_answer("broken", "scope") is None
        await redis.sadd(answer_cache._doc_index_key(3), key)
        assert await answer_cache.invalidate_by_document_ids([3]) == 1
        assert not await redis.exists(key)
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_optional_cache_logs_backend_failures_and_returns_miss(monkeypatch, caplog):
    redis = SimpleNamespace(
        get=AsyncMock(side_effect=OSError("offline")),
        pipeline=MagicMock(side_effect=OSError("offline")),
        smembers=AsyncMock(side_effect=OSError("offline")),
    )
    monkeypatch.setattr(answer_cache.redis_client, "_redis", redis)
    assert await answer_cache.find_cached_answer("q", "scope") is None
    await answer_cache.store_cached_answer("q", "q", "answer", [], "scope")
    assert await answer_cache.invalidate_by_document_ids([1]) == 0
    assert "Cache lookup failed" in caplog.text
    assert "Failed to store cached answer" in caplog.text
    assert "Cache invalidation failed" in caplog.text


@pytest.mark.asyncio
async def test_cache_hit_does_not_resurrect_concurrently_deleted_key(monkeypatch):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(answer_cache.redis_client, "_redis", redis)
    try:
        await answer_cache.store_cached_answer("q", "q", "answer", [], "scope", [1])
        original_get = redis.get
        key = answer_cache._cache_key("q", "scope")

        async def get_then_delete(cache_key):
            raw = await original_get(cache_key)
            await redis.delete(cache_key)
            return raw

        monkeypatch.setattr(redis, "get", get_then_delete)
        assert (await answer_cache.find_cached_answer("q", "scope"))["answer"] == "answer"
        assert not await redis.exists(key)
        assert await redis.ttl(key) == -2
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_cache_hit_preserves_expiration_and_removes_all_reverse_links(monkeypatch):
    """Exercise the Redis command contract without the native fake server parser."""
    import json
    import gzip

    values, indexes, ttls = {}, {}, {}

    async def get(key):
        return values.get(key)

    async def set_value(key, value, ex=None, keepttl=False, xx=False):
        if xx and key not in values:
            return None
        values[key] = value
        if ex is not None:
            ttls[key] = ex
        elif not keepttl:
            ttls.pop(key, None)
        return True

    async def members(key):
        return indexes.get(key, set())

    redis = SimpleNamespace(
        get=get,
        set=set_value,
        smembers=members,
        pipeline=lambda **kwargs: _CacheContractPipeline(values, indexes, ttls, get, set_value),
    )
    monkeypatch.setattr(answer_cache.redis_client, "_redis", redis)
    await answer_cache.store_cached_answer("Q", "q", "answer", [], "scope", [1, 2])
    key = answer_cache._cache_key("q", "scope")
    assert ttls[key] == answer_cache.CACHE_TTL_SECONDS
    assert isinstance(values[key], str)
    hit = await answer_cache.find_cached_answer("q", "scope")
    assert hit["answer"] == "answer" and hit["hit_count"] == 1
    assert ttls[key] == answer_cache.CACHE_TTL_SECONDS
    assert await answer_cache.invalidate_by_document_ids([1]) == 1
    assert not indexes[answer_cache._doc_index_key(2)]
    assert await answer_cache.find_cached_answer("q", "scope") is None
    # Binary entries from deployments using a binary Redis connection remain readable.
    legacy = gzip.compress(json.dumps({"answer": "legacy"}).encode())
    assert answer_cache._decode_entry(legacy)["answer"] == "legacy"
    assert answer_cache.compute_question_hash(
        "Q", {"date": "2026-01-01"}
    ) != answer_cache.compute_question_hash("Q", {"date": "2026-01-02"})


class _CacheContractPipeline:
    def __init__(self, values, indexes, ttls, get, set_value):
        self.commands = []
        self.values, self.indexes, self.ttls = values, indexes, ttls
        self._get, self._set_value = get, set_value

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self.commands.append((name, args, kwargs))
            return self

        return queue

    async def execute(self):
        values, indexes, ttls = self.values, self.indexes, self.ttls
        get, set_value = self._get, self._set_value
        result = []
        for name, args, kwargs in self.commands:
            if name == "set":
                result.append(await set_value(*args, **kwargs))
            elif name == "get":
                result.append(await get(*args))
            elif name == "sadd":
                indexes.setdefault(args[0], set()).add(args[1])
                result.append(1)
            elif name == "srem":
                indexes.setdefault(args[0], set()).discard(args[1])
                result.append(1)
            elif name == "expire":
                ttls[args[0]] = args[1]
                result.append(True)
            elif name == "delete":
                result.append(int(values.pop(args[0], None) is not None))
                ttls.pop(args[0], None)
            else:
                raise AssertionError(f"Unexpected Redis command: {name}")
        return result
