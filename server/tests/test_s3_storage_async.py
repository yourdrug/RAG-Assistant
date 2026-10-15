"""Native async S3 request, concurrency and lifecycle contracts."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError

from config import settings
from infrastructure.storage.file_storage import S3Storage


def client_harness(monkeypatch, *, max_concurrent=2):
    monkeypatch.setattr(settings, "s3_max_concurrent", max_concurrent, raising=False)
    monkeypatch.setattr(settings, "s3_bucket", "bucket")
    client = MagicMock()
    for name in ("head_object", "put_object", "delete_object", "copy_object", "get_object"):
        setattr(client, name, AsyncMock())
    context = SimpleNamespace(__aenter__=AsyncMock(return_value=client), __aexit__=AsyncMock())
    session = MagicMock(create_client=MagicMock(return_value=context))
    monkeypatch.setattr("infrastructure.storage.file_storage.get_session", lambda: session)
    return S3Storage(), client, context, session


@pytest.mark.asyncio
async def test_concurrent_lazy_init_one_client_and_idempotent_close(monkeypatch):
    storage, client, context, session = client_harness(monkeypatch)
    session.create_client.assert_not_called()
    await asyncio.gather(*(storage.upload_file(str(i), b"x") for i in range(10)))
    context.__aenter__.assert_awaited_once()
    session.create_client.assert_called_once()
    await asyncio.gather(storage.aclose(), storage.aclose())
    context.__aexit__.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closed"):
        await storage.upload_file("late", b"x")


@pytest.mark.asyncio
async def test_init_failure_discards_unentered_context_then_retries(monkeypatch):
    storage, client, context, _ = client_harness(monkeypatch)
    context.__aenter__.side_effect = [OSError("connect"), client]
    with pytest.raises(OSError, match="connect"):
        await storage.upload_file("a", b"x")
    context.__aexit__.assert_not_awaited()
    await storage.upload_file("a", b"x")
    await storage.aclose()
    context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound", "AccessDenied", "NoSuchBucket"])
async def test_head_maps_only_missing_key_to_none(monkeypatch, code):
    storage, client, _, _ = client_harness(monkeypatch)
    error = ClientError({"Error": {"Code": code}}, "HeadObject")
    client.head_object.side_effect = error
    try:
        if code in {"404", "NoSuchKey", "NotFound"}:
            assert await storage.get_file_info("нет.txt") is None
        else:
            with pytest.raises(ClientError) as caught:
                await storage.get_file_info("нет.txt")
            assert caught.value is error
    finally:
        await storage.aclose()


@pytest.mark.asyncio
async def test_list_pagination_sorting_and_request_arguments(monkeypatch):
    storage, client, _, _ = client_harness(monkeypatch)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    async def pages(**kwargs):
        assert kwargs == {"Bucket": "bucket", "Prefix": "docs/"}
        yield {"Contents": [{"Key": f"docs/{i:04}.txt", "Size": 1, "LastModified": now} for i in range(1000)]}
        yield {
            "Contents": [
                {"Key": "docs/отчёт с пробелом.PDF", "Size": 42, "LastModified": now},
                {"Key": "docs/skip.exe", "Size": 1, "LastModified": now},
            ]
        }

    client.get_paginator.return_value.paginate.side_effect = pages
    items = await storage.list_files("docs/")
    assert len(items) == 1001
    assert [i.key for i in items] == sorted(i.key for i in items)
    key = "docs/отчёт с пробелом.PDF"
    client.head_object.return_value = {"ContentLength": 42, "LastModified": now}
    info = await storage.get_file_info(key)
    assert (info.key, info.size_bytes, info.last_modified, info.extension) == (
        key,
        42,
        now.isoformat(),
        ".pdf",
    )
    await storage.upload_file(key, b"content")
    client.put_object.assert_awaited_once_with(Bucket="bucket", Key=key, Body=b"content")
    await storage.rename_file(key, "docs/new.pdf")
    client.copy_object.assert_awaited_once_with(
        Bucket="bucket", CopySource={"Bucket": "bucket", "Key": key}, Key="docs/new.pdf"
    )
    client.delete_object.assert_awaited_once_with(Bucket="bucket", Key=key)
    await storage.aclose()


@pytest.mark.asyncio
async def test_successful_rename_with_one_slot_copies_before_deleting(monkeypatch):
    storage, client, _, _ = client_harness(monkeypatch, max_concurrent=1)
    events = []

    async def copy(**kwargs):
        assert kwargs == {"Bucket": "bucket", "CopySource": {"Bucket": "bucket", "Key": "old"}, "Key": "new"}
        events.append("copy")

    async def delete(**kwargs):
        assert kwargs == {"Bucket": "bucket", "Key": "old"}
        events.append("delete")

    client.copy_object.side_effect = copy
    client.delete_object.side_effect = delete
    try:
        await asyncio.wait_for(storage.rename_file("old", "new"), 2)
        assert events == ["copy", "delete"]
    finally:
        await storage.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["AccessDenied", "NoSuchKey"])
async def test_copy_failure_never_deletes_source(monkeypatch, code):
    storage, client, _, _ = client_harness(monkeypatch, max_concurrent=1)
    client.copy_object.side_effect = ClientError({"Error": {"Code": code}}, "CopyObject")
    if code == "NoSuchKey":
        await asyncio.wait_for(storage.rename_file("missing", "new"), 2)
    else:
        with pytest.raises(ClientError):
            await storage.rename_file("old", "new")
    client.delete_object.assert_not_awaited()
    await storage.aclose()


@pytest.mark.asyncio
async def test_semaphore_cancel_and_shutdown_wait_for_active_operation(monkeypatch):
    storage, client, context, _ = client_harness(monkeypatch, max_concurrent=1)
    started, release = asyncio.Event(), asyncio.Event()

    async def put(**kwargs):
        started.set()
        await release.wait()

    client.put_object.side_effect = put
    first = asyncio.create_task(storage.upload_file("first", b"x"))
    await started.wait()
    second = asyncio.create_task(storage.upload_file("second", b"x"))
    await asyncio.sleep(0)
    assert client.put_object.await_count == 1
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    closing = asyncio.create_task(storage.aclose())
    await asyncio.sleep(0)
    context.__aexit__.assert_not_awaited()
    release.set()
    await asyncio.gather(first, closing)
    context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_unused_storage_does_not_initialize(monkeypatch):
    storage, _, _, session = client_harness(monkeypatch)
    await storage.aclose()
    session.create_client.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["head", "list", "get"])
async def test_delayed_network_keeps_heartbeat_responsive(monkeypatch, operation, record_property):
    from test_s3_storage_download import Body

    storage, client, _, _ = client_harness(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()
    if operation == "head":

        async def delayed_head(**kwargs):
            started.set()
            await release.wait()
            return {"ContentLength": 1, "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc)}

        client.head_object.side_effect = delayed_head
        call = storage.get_file_info("a.txt")
    elif operation == "list":

        async def delayed_pages(**kwargs):
            started.set()
            await release.wait()
            yield {"Contents": []}

        client.get_paginator.return_value.paginate.side_effect = delayed_pages
        call = storage.list_files("docs/")
    else:
        body = Body([])
        body.release = release
        started = body.started
        client.get_object.return_value = {"Body": body}
        call = storage.download_to_temp("a.txt")
    pending = asyncio.create_task(call)
    lags = []
    try:
        await asyncio.wait_for(started.wait(), 5)
        loop = asyncio.get_running_loop()
        until = loop.time() + 1
        while loop.time() < until:
            due = loop.time() + 0.05
            await asyncio.sleep(0.05)
            lags.append(max(0, loop.time() - due))
        assert not pending.done()
        assert len(lags) >= 10
        assert max(lags) < 0.2
        record_property("max_heartbeat_lag_ms", round(max(lags) * 1000, 3))
    finally:
        release.set()
    result = await pending
    if operation == "get":
        result.unlink()
    await storage.aclose()


@pytest.mark.asyncio
async def test_closed_storage_cannot_reopen_via_client_acquisition(monkeypatch):
    storage, _, _, session = client_harness(monkeypatch)
    await storage.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await storage.ensure_client()
    session.create_client.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError("S3 timeout"), asyncio.CancelledError()])
async def test_head_timeout_or_cancellation_propagates_and_releases_slot(monkeypatch, error):
    storage, client, _, _ = client_harness(monkeypatch, max_concurrent=1)
    client.head_object.side_effect = error
    with pytest.raises(type(error)):
        await storage.get_file_info("a.txt")
    await asyncio.wait_for(storage.upload_file("b.txt", b"content"), 2)
    client.put_object.assert_awaited_once()
    await storage.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_real_sdk_context_failed_creation_preserves_error_and_allows_retry(monkeypatch, cancelled):
    from aiobotocore.session import ClientCreatorContext

    storage, client, _, session = client_harness(monkeypatch)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock()
    started = asyncio.Event()
    failure = OSError("creation failed")
    attempts = 0

    async def create():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if cancelled:
                started.set()
                await asyncio.Event().wait()
            raise failure
        return client

    session.create_client.side_effect = lambda *args, **kwargs: ClientCreatorContext(create())
    if cancelled:
        pending = asyncio.create_task(storage.upload_file("a", b"x"))
        await asyncio.wait_for(started.wait(), 5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        with pytest.raises(OSError) as caught:
            await storage.upload_file("a", b"x")
        assert caught.value is failure
    client.__aexit__.assert_not_awaited()
    await storage.upload_file("a", b"x")
    await storage.aclose()
    client.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_multipart_bounded_and_aborted_on_error_or_cancel(monkeypatch):
    storage, client, _, _ = client_harness(monkeypatch)
    client.create_multipart_upload = AsyncMock(return_value={'UploadId': 'upload'})
    client.upload_part = AsyncMock(return_value={'ETag': 'etag'})
    client.complete_multipart_upload = AsyncMock()
    client.abort_multipart_upload = AsyncMock()
    monkeypatch.setattr(settings, 'data_dir', '/proc/unavailable')

    async def chunks():
        for _ in range(12):
            yield b'x' * 1024 * 1024

    await storage.upload_stream('benchmarks/runs/1/report.json', chunks())
    sizes = [len(call.kwargs['Body']) for call in client.upload_part.call_args_list]
    assert sizes == [5 * 1024 * 1024, 5 * 1024 * 1024, 2 * 1024 * 1024]
    client.complete_multipart_upload.assert_awaited_once()

    for error in (OSError('upload failed'), asyncio.CancelledError()):

        async def broken(error=error):
            yield b'partial'
            raise error

        with pytest.raises(type(error)):
            await storage.upload_stream('benchmarks/runs/1/report.json', broken())
    assert client.abort_multipart_upload.await_count == 2
    await storage.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("creation_fails", [False, True])
async def test_cancel_during_multipart_creation_aborts_remote_upload(monkeypatch, creation_fails):
    storage, client, _, _ = client_harness(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()

    async def create(**kwargs):
        started.set()
        await release.wait()
        if creation_fails:
            raise OSError('create failed')
        return {'UploadId': 'owned-upload'}

    client.create_multipart_upload = AsyncMock(side_effect=create)
    client.abort_multipart_upload = AsyncMock()

    async def blocks():
        yield b'data'

    task = asyncio.create_task(storage.upload_stream('report.json', blocks()))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    if creation_fails:
        client.abort_multipart_upload.assert_not_awaited()
    else:
        client.abort_multipart_upload.assert_awaited_once_with(
            Bucket='bucket', Key='report.json', UploadId='owned-upload'
        )
    await storage.aclose()
