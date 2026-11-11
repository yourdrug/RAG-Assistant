"""Opt-in S3 compatibility smoke (local stack: PGSTY Silo); only this run's unique keys are removed."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import resource
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker
from tests.integration.uow import session_factory

from application.benchmark import BenchmarkExportService
from config import settings
from domain.benchmark.run import BenchmarkRun
from domain.knowledge.enums import DocumentVisibility
from infrastructure.bm25 import bm25_invalidation, persistence
from infrastructure.bm25.snapshots import LocalBM25Snapshots
from infrastructure.redis.redis_client import redis_client
from infrastructure.repositories.benchmark.sqlalchemy_benchmark_run_repository import (
    SQLAlchemyBenchmarkRunRepository,
)
from infrastructure.storage.file_storage import LazyStorage, S3Storage


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine

    from infrastructure.bm25 import BM25Index


pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def s3_storage(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[S3Storage, str, list[str]]]:
    if os.environ.get("RAG_S3_SMOKE") != "1":
        pytest.skip("Set RAG_S3_SMOKE=1 and S3_SMOKE_ENDPOINT to opt in")

    endpoint = os.environ.get("S3_SMOKE_ENDPOINT")

    if not endpoint:
        pytest.fail("S3_SMOKE_ENDPOINT is required")

    monkeypatch.setattr(settings, "s3_endpoint", endpoint)
    storage = S3Storage(settings, settings)
    prefix = f"_test/async-s3/{uuid.uuid4().hex}/"
    keys: list[str] = []
    try:
        yield storage, prefix, keys
    finally:
        # Do not LIST/delete any existing objects or delete the bucket.
        try:
            results = await asyncio.gather(
                *(storage.delete_file(key) for key in keys), return_exceptions=True
            )
            errors = [result for result in results if isinstance(result, BaseException)]

            if errors:
                msg = "Test-prefix cleanup failed"
                raise RuntimeError(msg) from errors[0]
        finally:
            await storage.aclose()


@pytest.mark.asyncio
async def test_s3_operations_content_pagination_and_cleanup(
    s3_storage: tuple[S3Storage, str, list[str]], record_property: Callable[[str, object], None]
) -> None:
    storage, prefix, keys = s3_storage
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fd_before = open_fd_count()
    key = prefix + "отчёт с пробелом.txt"
    copied = prefix + "копия.txt"
    empty = prefix + "empty.txt"
    renamed = prefix + "renamed.txt"
    keys.extend([key, copied, empty, renamed])
    content = bytes(range(256)) * 12289  # Multiple 1 MiB blocks with a short tail.
    await storage.upload_file(key, content)
    info = await storage.get_file_info(key)
    assert info is not None
    assert (info.key, info.filename, info.size_bytes, info.extension) == (
        key,
        "отчёт с пробелом.txt",
        len(content),
        ".txt",
    )
    assert "T" in info.last_modified
    await assert_download(storage, key, content, repeats=5)
    await assert_copy_and_rename(storage, key, copied, renamed, len(content))
    await storage.upload_file(empty, b"")
    await assert_download(storage, empty, b"")
    assert await storage.get_file_info(prefix + "missing.txt") is None
    await assert_paginated_listing(storage, prefix, keys)
    await storage.delete_file(key)
    assert await storage.get_file_info(key) is None
    record_property("max_rss_before_kib", rss_before)
    record_property("max_rss_after_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    record_property("fds_before", fd_before)
    record_property("fds_after_downloads", open_fd_count())


@pytest.mark.asyncio
async def test_repeated_config_rotations_close_retired_pools(
    s3_storage: tuple[S3Storage, str, list[str]],
    record_property: Callable[[str, object], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, prefix, keys = s3_storage
    before = open_fd_count()
    storage = LazyStorage(settings, settings)
    key = prefix + "rotation.txt"
    keys.append(key)
    try:
        endpoints = [
            settings.s3_endpoint,
            os.environ.get("S3_SMOKE_ALTERNATE_ENDPOINT", settings.s3_endpoint),
        ]

        for iteration in range(5):
            monkeypatch.setattr(settings, "s3_endpoint", endpoints[iteration % len(endpoints)])
            await storage.upload_file(key, b"rotation")
            await assert_download(storage, key, b"rotation")
            storage.invalidate()

        record_property("rotation_fds_open", open_fd_count())
    finally:
        await storage.aclose()
    after = open_fd_count()
    record_property("rotation_fds_before", before)
    record_property("rotation_fds_after_close", after)
    assert after <= before + 2


@pytest.mark.asyncio
async def test_bm25_cold_warm_load_and_snapshot_refresh(
    s3_storage: tuple[S3Storage, str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    storage, prefix, keys = s3_storage
    key = prefix + "snapshot.json"
    keys.append(key)
    monkeypatch.setattr(persistence, "BM25_S3_KEY", key)
    downloads = []
    original_download = storage.download_to_temp

    async def download(snapshot_key: str) -> Path:
        downloads.append(snapshot_key)
        return await original_download(snapshot_key)

    monkeypatch.setattr(storage, "download_to_temp", download)
    registry = LocalBM25Snapshots(bm25_loader=partial(persistence.load_bm25_index_from_s3, storage))
    data = {
        "texts": ["original snapshot"],
        "hashes": ["original-hash"],
        "doc_visibility": [DocumentVisibility.INTERNAL_PRIVATE.value],
        "doc_owner_id": [42],
        "doc_group_id": [None],
    }
    try:
        await storage.upload_file(key, json.dumps(data).encode())
        snapshots = await asyncio.gather(*(registry.ensure_bm25_loaded() for _ in range(10)))
        first = snapshots[0]
        assert_cold_snapshot(snapshots, first, downloads, key)
        assert await registry.ensure_bm25_loaded() is first
        data["texts"] = ["replacement snapshot"]
        data["hashes"] = ["replacement-hash"]
        await storage.upload_file(key, json.dumps(data).encode())
        registry.invalidate_bm25()
        second = await registry.ensure_bm25_loaded()
        assert second is not first
        assert second is not None
        assert second.hashes == ["replacement-hash"]
        assert downloads == [key, key]
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_bm25_redis_invalidation_reloads_real_snapshot(
    s3_storage: tuple[S3Storage, str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.environ.get("RAG_S3_REDIS_SMOKE") != "1":
        pytest.skip("Set RAG_S3_REDIS_SMOKE=1 to opt in to isolated Redis pub/sub smoke")

    storage, prefix, keys = s3_storage
    key = prefix + "redis-snapshot.json"
    keys.append(key)
    channel = prefix + "invalidate"
    monkeypatch.setattr(persistence, "BM25_S3_KEY", key)
    monkeypatch.setattr(bm25_invalidation, "_CHANNEL", channel)
    registry = LocalBM25Snapshots(bm25_loader=partial(persistence.load_bm25_index_from_s3, storage))
    listener = None
    try:
        await redis_client.init()
        await storage.upload_file(key, b'{"texts":["before"],"hashes":["before-hash"]}')
        first = await registry.ensure_bm25_loaded()
        listener = asyncio.create_task(bm25_invalidation.listen_for_bm25_invalidation(registry))
        await storage.upload_file(key, b'{"texts":["after"],"hashes":["after-hash"]}')

        second = await asyncio.wait_for(wait_for_reload(registry, first), 5)
        assert second is not None
        assert second.hashes == ["after-hash"]
    finally:
        if listener is not None:
            listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener

        await registry.close()
        await redis_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("report_format", ["csv", "json"])
async def test_committed_benchmark_report_survives_s3_failure_and_new_worker(
    pg_engine: AsyncEngine,
    s3_storage: tuple[S3Storage, str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    report_format: str,
) -> None:
    storage, prefix, keys = s3_storage
    monkeypatch.setattr(settings, "data_dir", "/proc/unavailable-benchmark-data")
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    results = [{"id": n, "question": "question", "answer": "x" * 100_000} for n in range(60)]
    async with sessions.begin() as session:
        run = await SQLAlchemyBenchmarkRunRepository(session).create(
            BenchmarkRun(per_question_results=results)
        )
        assert run.id is not None

    async def upload(key: str, blocks: AsyncIterator[bytes]) -> None:
        # Prefix every object, including the service's stable report key, for safe cleanup.
        object_key = prefix + key

        if object_key not in keys:
            keys.append(object_key)

        await storage.upload_stream(object_key, blocks)

    factory = session_factory(sessions)
    reports = SimpleNamespace(upload_stream=upload)
    with monkeypatch.context() as failed_storage:
        failed_storage.setattr(storage, "bucket", "missing-" + uuid.uuid4().hex)
        assert not await BenchmarkExportService(factory, reports).save_report(run.id, report_format)
    async with sessions() as session:
        saved = await SQLAlchemyBenchmarkRunRepository(session).get_by_id(run.id)
        assert saved is not None
        assert saved.per_question_results == results
        assert saved.artifacts[report_format]["status"] == "failed"

    # A new service/session factory simulates another worker retrying the committed run.
    assert await BenchmarkExportService(session_factory(sessions), reports).save_report(run.id, report_format)
    async with sessions() as session:
        saved = await SQLAlchemyBenchmarkRunRepository(session).get_by_id(run.id)
        assert saved is not None
        metadata = saved.artifacts[report_format]
    assert len(keys) == 1
    await assert_report(storage, prefix, metadata, results, report_format)


def open_fd_count() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


async def assert_download(
    storage: S3Storage | LazyStorage, key: str, content: bytes, *, repeats: int = 1
) -> None:
    for _ in range(repeats):
        path = await storage.download_to_temp(key)
        try:
            downloaded = await asyncio.to_thread(path.read_bytes)
            assert hashlib.sha256(downloaded).digest() == hashlib.sha256(content).digest()
            assert downloaded == content
        finally:
            await asyncio.to_thread(path.unlink)


async def assert_paginated_listing(storage: S3Storage, prefix: str, keys: list[str]) -> None:
    pagination_keys = [prefix + f"page/{i:04}.txt" for i in range(1001)]
    keys.extend(pagination_keys)
    await asyncio.gather(*(storage.upload_file(name, b"x") for name in pagination_keys))
    listing = await storage.list_files(prefix + "page/")
    assert [item.key for item in listing] == pagination_keys
    assert all(item.size_bytes == 1 for item in listing)


def assert_cold_snapshot(
    snapshots: list[BM25Index | None], first: BM25Index | None, downloads: list[str], key: str
) -> None:
    assert all(snapshot is first for snapshot in snapshots)
    assert downloads == [key]
    assert first is not None
    assert first.doc_visibility == [DocumentVisibility.INTERNAL_PRIVATE.value]
    assert first.doc_owner_id == [42]


async def wait_for_reload(registry: LocalBM25Snapshots, first: BM25Index | None) -> BM25Index:
    while True:
        # A unique channel keeps running server/worker caches untouched.
        await bm25_invalidation.publish_bm25_invalidation()
        await asyncio.sleep(0.05)
        current = await registry.ensure_bm25_loaded()

        if current is not None and current is not first:
            return current


async def assert_report(
    storage: S3Storage,
    prefix: str,
    metadata: dict[str, Any],
    results: list[dict[str, Any]],
    report_format: str,
) -> None:
    assert metadata["status"] == "ready"
    assert metadata["size_bytes"] > 5 * 1024 * 1024
    key = prefix + metadata["key"]
    info = await storage.get_file_info(key)
    assert info is not None
    assert info.size_bytes == metadata["size_bytes"]
    path = await storage.download_to_temp(key)
    try:
        assert (await asyncio.to_thread(path.stat)).st_size == metadata["size_bytes"]

        if report_format == "json":
            assert json.loads(await asyncio.to_thread(path.read_bytes))["results"] == results
    finally:
        await asyncio.to_thread(path.unlink)


async def assert_copy_and_rename(storage: S3Storage, key: str, copied: str, renamed: str, size: int) -> None:
    await storage.copy_file(key, copied)
    await storage.rename_file(copied, renamed)
    assert await storage.get_file_info(copied) is None
    renamed_info = await storage.get_file_info(renamed)
    assert renamed_info is not None
    assert renamed_info.size_bytes == size
