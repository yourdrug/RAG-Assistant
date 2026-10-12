"""Transitional thread-only BM25 reader; removed by migration stage B."""

import asyncio
import threading

import boto3

from config import settings
from infrastructure.storage.file_storage import s3_client_kwargs


class S3SnapshotReader:
    """Own one synchronous client with a fixed config snapshot."""

    def __init__(self):
        self.bucket = settings.s3_bucket
        self.enabled = settings.file_backend == "s3"
        self._kwargs = s3_client_kwargs()
        self._client = None
        self._closed = False
        self._lock = threading.Lock()

    def download_bytes(self, key: str) -> bytes:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("BM25 snapshot reader must run in a worker thread")
        with self._lock:
            if self._closed:
                raise RuntimeError("BM25 snapshot reader is closed")
            if self._client is None:
                self._client = boto3.client("s3", **self._kwargs)
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            try:
                return response["Body"].read()
            finally:
                response["Body"].close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._client is not None:
                try:
                    self._client.close()
                finally:
                    self._client = None
