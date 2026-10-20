"""Failure recovery for batch uploads through the shared ingestion service."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from application.ingestion.targets import S3IngestionTargets, S3UploadService


@pytest.fixture
def storage() -> MagicMock:
    objects: dict[str, bytes] = {}

    async def upload(key: str, data: bytes) -> None:
        objects[key] = data

    async def delete(key: str) -> None:
        objects.pop(key, None)

    async def copy(old_key: str, new_key: str) -> None:
        objects[new_key] = objects[old_key]

    return MagicMock(
        objects=objects,
        upload_file=AsyncMock(side_effect=upload),
        delete_file=AsyncMock(side_effect=delete),
        copy_file=AsyncMock(side_effect=copy),
        get_file_info=AsyncMock(side_effect=lambda key: object() if key in objects else None),
    )


def files(*names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(filename=name, data=name.encode()) for name in names]


def fail_upload(storage: MagicMock, failed_key: str, error: BaseException) -> None:
    upload = storage.upload_file.side_effect

    async def upload_then_fail(key: str, data: bytes) -> None:
        await upload(key, data)

        if key == failed_key:
            raise error

    storage.upload_file.side_effect = upload_then_fail


@pytest.mark.asyncio
async def test_success_returns_keys_and_cleans_backups(storage: MagicMock) -> None:
    storage.objects["docs/a.pdf"] = b"old"
    service = S3UploadService(storage, S3IngestionTargets())

    assert await service.upload_files(files("a.pdf", "b.pdf")) == ["docs/a.pdf", "docs/b.pdf"]
    assert storage.objects == {"docs/a.pdf": b"a.pdf", "docs/b.pdf": b"b.pdf"}


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [OSError("upload failed"), asyncio.CancelledError()])
@pytest.mark.parametrize("existing", [False, True])
async def test_failure_restores_original_batch_state(
    storage: MagicMock, error: BaseException, existing: bool
) -> None:
    if existing:
        storage.objects["docs/a.pdf"] = b"old"
        storage.objects["docs/b.pdf"] = b"old b"

    original = dict(storage.objects)
    fail_upload(storage, "docs/b.pdf", error)
    service = S3UploadService(storage, S3IngestionTargets())

    with pytest.raises(type(error)) as raised:
        await service.upload_files(files("a.pdf", "b.pdf", "c.pdf"))

    assert raised.value is error
    assert storage.objects == original


@pytest.mark.asyncio
async def test_invalid_key_is_rejected_before_any_storage_write(storage: MagicMock) -> None:
    service = S3UploadService(storage, S3IngestionTargets())

    with pytest.raises(ValueError, match="S3 key"):
        await service.upload_files(files("a.pdf", "../b.pdf"))

    storage.upload_file.assert_not_awaited()
    storage.copy_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_key_rolls_back_to_original_content(storage: MagicMock) -> None:
    storage.objects["docs/a.pdf"] = b"old"
    fail_upload(storage, "docs/c.pdf", OSError("upload failed"))
    service = S3UploadService(storage, S3IngestionTargets())

    with pytest.raises(OSError, match="upload failed"):
        await service.upload_files(files("a.pdf", "a.pdf", "c.pdf"))

    assert storage.objects == {"docs/a.pdf": b"old"}
    assert storage.copy_file.await_count == 2


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_stop_remaining_compensation(
    storage: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    fail_upload(storage, "docs/c.pdf", OSError("upload failed"))
    delete = storage.delete_file.side_effect

    async def fail_one_delete(key: str) -> None:
        if key == "docs/b.pdf":
            msg = "delete failed"
            raise OSError(msg)

        await delete(key)

    storage.delete_file.side_effect = fail_one_delete
    service = S3UploadService(storage, S3IngestionTargets())

    with pytest.raises(OSError, match="upload failed"):
        await service.upload_files(files("a.pdf", "b.pdf", "c.pdf"))

    assert "Failed to roll back batch upload" in caplog.text
    assert storage.delete_file.await_count == 3
    assert storage.objects == {"docs/b.pdf": b"b.pdf"}


@pytest.mark.asyncio
async def test_failed_restore_keeps_backup_for_recovery(
    storage: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    storage.objects["docs/a.pdf"] = b"old"
    fail_upload(storage, "docs/b.pdf", OSError("upload failed"))
    copy = storage.copy_file.side_effect

    async def fail_restore(old_key: str, new_key: str) -> None:
        if new_key == "docs/a.pdf":
            msg = "restore failed"
            raise OSError(msg)

        await copy(old_key, new_key)

    storage.copy_file.side_effect = fail_restore
    service = S3UploadService(storage, S3IngestionTargets())

    with pytest.raises(OSError, match="upload failed"):
        await service.upload_files(files("a.pdf", "b.pdf"))

    backup_keys = set(storage.objects) - {"docs/a.pdf"}
    assert len(backup_keys) == 1
    assert storage.objects[backup_keys.pop()] == b"old"
    assert "Failed to roll back batch upload" in caplog.text


@pytest.mark.asyncio
async def test_failed_backup_does_not_change_original_objects(storage: MagicMock) -> None:
    storage.objects["docs/a.pdf"] = b"old"
    copy = storage.copy_file.side_effect
    error = OSError("backup failed")

    async def copy_then_fail(old_key: str, new_key: str) -> None:
        await copy(old_key, new_key)
        raise error

    storage.copy_file.side_effect = copy_then_fail
    service = S3UploadService(storage, S3IngestionTargets())

    with pytest.raises(OSError, match="backup failed"):
        await service.upload_files(files("a.pdf", "b.pdf"))

    storage.upload_file.assert_not_awaited()
    assert storage.objects == {"docs/a.pdf": b"old"}


@pytest.mark.asyncio
async def test_backup_cleanup_failure_is_logged_after_success(
    storage: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    storage.objects["docs/a.pdf"] = b"old"
    storage.delete_file.side_effect = OSError("delete failed")
    service = S3UploadService(storage, S3IngestionTargets())

    assert await service.upload_files(files("a.pdf")) == ["docs/a.pdf"]
    assert storage.objects["docs/a.pdf"] == b"a.pdf"
    assert "Failed to delete upload backup" in caplog.text
