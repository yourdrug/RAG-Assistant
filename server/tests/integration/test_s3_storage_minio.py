"""Opt-in MinIO compatibility smoke; only this run's unique keys are removed."""

import asyncio
import hashlib
import os
import resource
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

from config import settings
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
async def test_repeated_config_rotations_close_retired_pools(minio_storage, record_property):
    _, prefix, keys = minio_storage
    before = len(list(Path("/proc/self/fd").iterdir()))
    storage = LazyStorage()
    key = prefix + "rotation.txt"
    keys.append(key)
    try:
        for _ in range(5):
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
