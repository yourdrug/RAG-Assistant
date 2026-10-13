"""Opt-in MinIO compatibility smoke; only this run's unique keys are removed."""

import asyncio
import hashlib
import json
import os
import resource
import uuid
from contextlib import suppress
from functools import partial
from pathlib import Path

import pytest
import pytest_asyncio

from config import settings
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.bm25 import bm25_invalidation, persistence
from infrastructure.ml.clients.client_registry import MLClientRegistry
from infrastructure.redis.redis_client import redis_client
from infrastructure.storage.file_storage import S3Storage, LazyStorage

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def minio_storage(monkeypatch):
    if os.environ.get("RAG_S3_SMOKE") != "1":
        pytest.skip("Set RAG_S3_SMOKE=1 and S3_SMOKE_ENDPOINT to opt in")
    endpoint = os.environ.get("S3_SMOKE_ENDPOINT")
    if not endpoint:
        pytest.fail("S3_SMOKE_ENDPOINT is required")
    monkeypatch.setattr(settings, "s3_endpoint", endpoint)
    monkeypatch.setattr(settings, "file_backend", "s3")
    storage = S3Storage()
    prefix = f"_test/async-s3/{uuid.uuid4().hex}/"
    keys = []
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
                raise RuntimeError("Test-prefix cleanup failed") from errors[0]
        finally:
            await storage.aclose()


@pytest.mark.asyncio
async def test_minio_operations_content_pagination_and_cleanup(minio_storage, record_property):
    storage, prefix, keys = minio_storage
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fd_before = len(list(Path("/proc/self/fd").iterdir()))
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
    for _ in range(5):
        path = await storage.download_to_temp(key)
        try:
            assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(content).digest()
        finally:
            path.unlink()
    await storage.copy_file(key, copied)
    await storage.rename_file(copied, renamed)
    assert await storage.get_file_info(copied) is None
    assert (await storage.get_file_info(renamed)).size_bytes == len(content)
    await storage.upload_file(empty, b"")
    path = await storage.download_to_temp(empty)
    try:
        assert path.read_bytes() == b""
    finally:
        path.unlink()
    assert await storage.get_file_info(prefix + "missing.txt") is None
    pagination_keys = [prefix + f"page/{i:04}.txt" for i in range(1001)]
    keys.extend(pagination_keys)
    await asyncio.gather(*(storage.upload_file(name, b"x") for name in pagination_keys))
    listing = await storage.list_files(prefix + "page/")
    assert [item.key for item in listing] == pagination_keys
    assert all(item.size_bytes == 1 for item in listing)
    await storage.delete_file(key)
    assert await storage.get_file_info(key) is None
    record_property("max_rss_before_kib", rss_before)
    record_property("max_rss_after_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    record_property("fds_before", fd_before)
    record_property("fds_after_downloads", len(list(Path("/proc/self/fd").iterdir())))


@pytest.mark.asyncio
async def test_repeated_config_rotations_close_retired_pools(minio_storage, record_property, monkeypatch):
    _, prefix, keys = minio_storage
    before = len(list(Path("/proc/self/fd").iterdir()))
    storage = LazyStorage()
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
            path = await storage.download_to_temp(key)
            try:
                assert path.read_bytes() == b"rotation"
            finally:
                path.unlink()
            storage.invalidate()
        record_property("rotation_fds_open", len(list(Path("/proc/self/fd").iterdir())))
    finally:
        await storage.aclose()
    after = len(list(Path("/proc/self/fd").iterdir()))
    record_property("rotation_fds_before", before)
    record_property("rotation_fds_after_close", after)
    assert after <= before + 2


@pytest.mark.asyncio
async def test_bm25_cold_warm_load_and_snapshot_refresh(minio_storage, monkeypatch):
    storage, prefix, keys = minio_storage
    key = prefix + "snapshot.json"
    keys.append(key)
    monkeypatch.setattr(persistence, "BM25_S3_KEY", key)
    downloads = []
    original_download = storage.download_to_temp

    async def download(snapshot_key):
        downloads.append(snapshot_key)
        return await original_download(snapshot_key)

    monkeypatch.setattr(storage, "download_to_temp", download)
    registry = MLClientRegistry(bm25_loader=partial(persistence.load_bm25_index_from_s3, storage))
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
        assert all(snapshot is first for snapshot in snapshots)
        assert await registry.ensure_bm25_loaded() is first
        assert downloads == [key]
        assert first.doc_visibility == [DocumentVisibility.INTERNAL_PRIVATE.value]
        assert first.doc_owner_id == [42]
        data["texts"] = ["replacement snapshot"]
        data["hashes"] = ["replacement-hash"]
        await storage.upload_file(key, json.dumps(data).encode())
        registry.invalidate_bm25()
        second = await registry.ensure_bm25_loaded()
        assert second is not first
        assert second.hashes == ["replacement-hash"]
        assert downloads == [key, key]
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_bm25_redis_invalidation_reloads_real_snapshot(minio_storage, monkeypatch):
    if os.environ.get("RAG_S3_REDIS_SMOKE") != "1":
        pytest.skip("Set RAG_S3_REDIS_SMOKE=1 to opt in to isolated Redis pub/sub smoke")
    storage, prefix, keys = minio_storage
    key = prefix + "redis-snapshot.json"
    keys.append(key)
    channel = prefix + "invalidate"
    monkeypatch.setattr(persistence, "BM25_S3_KEY", key)
    monkeypatch.setattr(bm25_invalidation, "_CHANNEL", channel)
    registry = MLClientRegistry(bm25_loader=partial(persistence.load_bm25_index_from_s3, storage))
    listener = None
    try:
        await redis_client.init()
        await storage.upload_file(key, b'{"texts":["before"],"hashes":["before-hash"]}')
        first = await registry.ensure_bm25_loaded()
        listener = asyncio.create_task(bm25_invalidation.listen_for_bm25_invalidation(registry))
        await storage.upload_file(key, b'{"texts":["after"],"hashes":["after-hash"]}')

        async def wait_for_reload():
            while True:
                # A unique channel keeps running server/worker caches untouched.
                await bm25_invalidation.publish_bm25_invalidation()
                await asyncio.sleep(0.05)
                current = await registry.ensure_bm25_loaded()
                if current is not first:
                    return current

        second = await asyncio.wait_for(wait_for_reload(), 5)
        assert second.hashes == ["after-hash"]
    finally:
        if listener is not None:
            listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener
        await registry.close()
        await redis_client.aclose()
