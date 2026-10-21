"""Container-owned async S3 storage with lazy client initialization."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any
from pathlib import Path

from aiobotocore.session import get_session
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from application.ports.file_storage import FileItem, FileStorage
from config import settings

log = logging.getLogger("default")
S3_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})

S3_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def s3_client_kwargs() -> dict[str, Any]:
    """Snapshot credentials/config for a single backend lifetime."""
    custom_endpoint = settings.s3_endpoint and not settings.s3_endpoint.startswith("https://s3.amazonaws.com")
    kwargs: dict[str, Any] = {
        "aws_access_key_id": settings.s3_access_key,
        "aws_secret_access_key": settings.s3_secret_key,
        "region_name": settings.s3_region,
        "config": BotoConfig(
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 3, "mode": "standard"},
            max_pool_connections=settings.s3_max_concurrent,
            s3={"addressing_style": "path"} if custom_endpoint else {},
        ),
    }
    if custom_endpoint:
        kwargs["endpoint_url"] = settings.s3_endpoint
    return kwargs


async def finish_owned_task(task: asyncio.Task):
    """Drain owned work even when the caller is cancelled."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def run_storage_io(function, *args, **kwargs):
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as exc:
        try:
            await finish_owned_task(task)
        finally:
            # A failed thread job must not replace the caller's cancellation.
            raise exc


def write_download_block(handle, block: bytes) -> None:
    handle.write(block)


def close_download(handle, *, remove: bool) -> None:
    try:
        handle.close()
    finally:
        if remove:
            Path(handle.name).unlink(missing_ok=True)


class S3Storage:
    """One lazy async S3 client, owned by one container and one event loop."""

    def __init__(self):
        self.bucket = settings.s3_bucket
        self._kwargs = s3_client_kwargs()
        self._extensions = settings.supported_extensions
        self._limit = asyncio.Semaphore(settings.s3_max_concurrent)
        self._acquire_timeout = settings.semaphore_acquire_timeout
        self._init_lock = asyncio.Lock()
        self._client = None
        self._context = None
        self._loop = None
        self._closing = False
        self._close_task = None
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return self._extensions

    def check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("S3 storage belongs to another event loop")

    async def ensure_client(self):
        """Acquire a client through the same admission/teardown boundary as requests."""
        async with self.operation() as client:
            return client

    async def initialize_client(self):
        if not self._active:
            raise RuntimeError("S3 client initialization requires an active storage operation")
        async with self._init_lock:
            if self._client is None:
                context = get_session().create_client("s3", **self._kwargs)
                # Like async with, exit only a successfully entered SDK context.
                # ClientCreatorContext has no client to close if creation fails.
                client = await context.__aenter__()
                self._context = context
                self._client = client
            return self._client

    @asynccontextmanager
    async def operation(self):
        self.check_loop()
        if self._closing:
            raise RuntimeError("S3 storage is closed")
        self._active += 1
        self._idle.clear()
        acquired = False
        try:
            await asyncio.wait_for(self._limit.acquire(), timeout=self._acquire_timeout)
            acquired = True
            yield await self.initialize_client()
        finally:
            if acquired:
                self._limit.release()
            self._active -= 1
            if not self._active:
                self._idle.set()

    async def list_files(self, prefix: str) -> list[FileItem]:
        items: list[FileItem] = []
        async with self.operation() as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    extension = Path(key).suffix.lower()
                    if extension in self._extensions:
                        items.append(
                            FileItem(
                                key, Path(key).name, obj["Size"], obj["LastModified"].isoformat(), extension
                            )
                        )
        return sorted(items, key=lambda item: item.key)

    async def get_file_info(self, key: str) -> FileItem | None:
        async with self.operation() as client:
            try:
                response = await client.head_object(Bucket=self.bucket, Key=key)
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in S3_MISSING_CODES:
                    return None
                raise
        return FileItem(
            key,
            Path(key).name,
            response["ContentLength"],
            response["LastModified"].isoformat(),
            Path(key).suffix.lower(),
        )

    async def download_to_temp(self, key: str) -> Path:
        async with self.operation() as client:
            response = await client.get_object(Bucket=self.bucket, Key=key)
            async with response["Body"] as body:
                # Shield temp creation too: cancellation must not lose its path.
                creation = asyncio.create_task(
                    asyncio.to_thread(tempfile.NamedTemporaryFile, delete=False, suffix=Path(key).suffix)
                )
                try:
                    handle = await asyncio.shield(creation)
                except asyncio.CancelledError as exc:
                    try:
                        handle = await finish_owned_task(creation)
                        await run_storage_io(close_download, handle, remove=True)
                    except OSError:
                        log.warning("Failed to create/clean up S3 download after cancellation", exc_info=True)
                    finally:
                        # Neither thread failure nor cleanup may replace cancellation.
                        raise exc
                except BaseException:
                    handle = await finish_owned_task(creation)
                    await run_storage_io(close_download, handle, remove=True)
                    raise
                try:
                    while block := await body.read(S3_DOWNLOAD_CHUNK_BYTES):
                        await run_storage_io(write_download_block, handle, block)
                    await run_storage_io(close_download, handle, remove=False)
                except asyncio.CancelledError as exc:
                    try:
                        await run_storage_io(close_download, handle, remove=True)
                    except OSError:
                        log.warning("Failed to clean up S3 download after cancellation", exc_info=True)
                    finally:
                        raise exc
                except BaseException:
                    await run_storage_io(close_download, handle, remove=True)
                    raise
                return Path(handle.name)

    async def upload_file(self, key: str, data: bytes) -> None:
        async with self.operation() as client:
            await client.put_object(Bucket=self.bucket, Key=key, Body=data)

    async def upload_stream(self, key: str, blocks: AsyncIterator[bytes]) -> None:
        # Multipart storage keeps memory bounded without a persistent local spool.
        async with self.operation() as client:
            creation = asyncio.create_task(client.create_multipart_upload(Bucket=self.bucket, Key=key))
            try:
                created = await asyncio.shield(creation)
            except asyncio.CancelledError as exc:
                try:
                    created = await finish_owned_task(creation)
                    abort = asyncio.create_task(
                        client.abort_multipart_upload(
                            Bucket=self.bucket, Key=key, UploadId=created["UploadId"]
                        )
                    )
                    await finish_owned_task(abort)
                except Exception:
                    log.exception("Multipart creation/cleanup failed after cancellation")
                finally:
                    raise exc
            upload_id = created["UploadId"]
            parts: list[dict] = []
            buffer = bytearray()
            part_size = 5 * 1024 * 1024
            try:

                async def send_part(data):
                    number = len(parts) + 1
                    response = await client.upload_part(
                        Bucket=self.bucket, Key=key, UploadId=upload_id, PartNumber=number, Body=data
                    )
                    parts.append({"PartNumber": number, "ETag": response["ETag"]})

                async for block in blocks:
                    for start in range(0, len(block), part_size):
                        buffer.extend(block[start : start + part_size])
                        if len(buffer) >= part_size:
                            await send_part(bytes(buffer[:part_size]))
                            del buffer[:part_size]
                if buffer or not parts:
                    await send_part(bytes(buffer))
                await client.complete_multipart_upload(
                    Bucket=self.bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": parts}
                )
            except BaseException:
                abort = asyncio.create_task(
                    client.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)
                )
                try:
                    await finish_owned_task(abort)
                except Exception:
                    log.exception("Failed to abort report upload")
                raise

    async def delete_file(self, key: str) -> None:
        async with self.operation() as client:
            await client.delete_object(Bucket=self.bucket, Key=key)

    async def copy_with_client(self, client, old_key: str, new_key: str) -> bool:
        try:
            await client.copy_object(
                Bucket=self.bucket, CopySource={"Bucket": self.bucket, "Key": old_key}, Key=new_key
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "NoSuchKey":
                log.warning("S3 copy: source key '%s' does not exist, skipping", old_key)
                return False
            raise
        return True

    async def copy_file(self, old_key: str, new_key: str) -> None:
        async with self.operation() as client:
            await self.copy_with_client(client, old_key, new_key)

    async def rename_file(self, old_key: str, new_key: str) -> None:
        # One slot and one backend for the complete operation, including shutdown.
        async with self.operation() as client:
            if await self.copy_with_client(client, old_key, new_key):
                await client.delete_object(Bucket=self.bucket, Key=old_key)

    async def close_client(self) -> None:
        await self._idle.wait()
        if self._context is not None:
            try:
                await self._context.__aexit__(None, None, None)
            finally:
                self._context = None
                self._client = None

    async def aclose(self) -> None:
        self.check_loop()
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self.close_client())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await finish_owned_task(self._close_task)
            raise


class LazyStorage:
    """Container-owned backend rotation; in-flight calls retain their backend."""

    def __init__(self):
        self._resolved = None
        self._retired = []
        self._lock = threading.Lock()
        self._closed = False
        self._close_task = None

    def ensure(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Storage is closed")
            if self._resolved is None:
                self._resolved = create_storage()
            return self._resolved

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return self.ensure().supported_extensions

    async def list_files(self, prefix: str) -> list[FileItem]:
        return await self.ensure().list_files(prefix)

    async def get_file_info(self, key: str) -> FileItem | None:
        return await self.ensure().get_file_info(key)

    async def download_to_temp(self, key: str) -> Path:
        return await self.ensure().download_to_temp(key)

    async def upload_file(self, key: str, data: bytes) -> None:
        await self.ensure().upload_file(key, data)

    async def upload_stream(self, key: str, blocks: AsyncIterator[bytes]) -> None:
        await self.ensure().upload_stream(key, blocks)

    async def delete_file(self, key: str) -> None:
        await self.ensure().delete_file(key)

    async def copy_file(self, old_key: str, new_key: str) -> None:
        await self.ensure().copy_file(old_key, new_key)

    async def rename_file(self, old_key: str, new_key: str) -> None:
        await self.ensure().rename_file(old_key, new_key)

    def invalidate(self) -> None:
        with self._lock:
            if self._resolved is not None:
                self._retired.append(self._resolved)
                self._resolved = None

    async def close_backends(self, backends) -> None:
        error = None
        for backend in backends:
            try:
                await backend.aclose()
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    async def aclose(self) -> None:
        with self._lock:
            if self._close_task is None:
                self._closed = True
                backends = [*self._retired]
                if self._resolved is not None:
                    backends.append(self._resolved)
                self._retired.clear()
                self._resolved = None
                self._close_task = asyncio.create_task(self.close_backends(backends))
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await finish_owned_task(self._close_task)
            raise


def create_storage() -> FileStorage:
    return S3Storage()
