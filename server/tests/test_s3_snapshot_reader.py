"""The transitional reader is thread-only, owned and cleans response bodies."""

import asyncio
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from config import settings
from infrastructure.bm25.s3_snapshot_reader import S3SnapshotReader
from infrastructure.bm25.persistence import load_bm25_index_from_s3_sync, BM25LoadError


@pytest.mark.asyncio
async def test_reader_reuses_and_closes_client_outside_loop(monkeypatch):
    client = MagicMock()
    client.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=b"content"))}
    create = MagicMock(return_value=client)
    monkeypatch.setattr("infrastructure.bm25.s3_snapshot_reader.boto3.client", create)
    monkeypatch.setattr(settings, "file_backend", "s3")
    reader = S3SnapshotReader()
    create.assert_not_called()
    with pytest.raises(RuntimeError, match="thread"):
        reader.download_bytes("snapshot")
    assert await asyncio.to_thread(reader.download_bytes, "snapshot") == b"content"
    assert await asyncio.to_thread(reader.download_bytes, "snapshot") == b"content"
    create.assert_called_once()
    assert client.get_object.return_value["Body"].close.call_count == 2
    await asyncio.to_thread(reader.close)
    await asyncio.to_thread(reader.close)
    client.close.assert_called_once()


@pytest.mark.parametrize("code", ["NoSuchKey", "AccessDenied", "NoSuchBucket"])
def test_missing_snapshot_distinguished_from_failed_load(code):
    reader = MagicMock()
    reader.download_bytes.side_effect = ClientError({"Error": {"Code": code}}, "GetObject")
    if code == "NoSuchKey":
        assert load_bm25_index_from_s3_sync(reader) is None
    else:
        with pytest.raises(BM25LoadError) as caught:
            load_bm25_index_from_s3_sync(reader)
        assert caught.value.__cause__ is reader.download_bytes.side_effect


def test_corrupt_snapshot_is_retryable_load_failure():
    with pytest.raises(BM25LoadError):
        load_bm25_index_from_s3_sync(MagicMock(download_bytes=MagicMock(return_value=b"bad json")))


def test_network_snapshot_failure_is_wrapped_and_can_retry():
    from botocore.exceptions import EndpointConnectionError

    reader = MagicMock()
    failure = EndpointConnectionError(endpoint_url="http://unavailable")
    reader.download_bytes.side_effect = failure
    with pytest.raises(BM25LoadError) as caught:
        load_bm25_index_from_s3_sync(reader)
    assert caught.value.__cause__ is failure
