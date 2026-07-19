"""
infrastructure/storage.py — File storage abstraction (local filesystem / S3).
"""

import functools
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from config import settings

log = logging.getLogger("default")


@dataclass
class FileItem:
    key: str
    filename: str
    size_bytes: int
    last_modified: str  # ISO format
    extension: str


@runtime_checkable
class FileStorage(Protocol):
    def list_files(self, prefix: str) -> list[FileItem]: ...
    def download_to_temp(self, key: str) -> Path: ...
    def upload_file(self, key: str, data: bytes) -> None: ...
    def get_file_info(self, key: str) -> FileItem | None: ...
    def delete_file(self, key: str) -> None: ...


class LocalStorage:
    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or settings.data_dir)

    def list_files(self, prefix: str) -> list[FileItem]:
        base = self.base_dir / prefix
        if not base.exists():
            return []
        items: list[FileItem] = []
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() not in settings.supported_extensions:
                continue
            rel = f.relative_to(self.base_dir)
            stat = f.stat()
            items.append(
                FileItem(
                    key=str(rel),
                    filename=f.name,
                    size_bytes=stat.st_size,
                    last_modified=str(int(stat.st_mtime)),
                    extension=f.suffix.lower(),
                )
            )
        return items

    def download_to_temp(self, key: str) -> Path:
        src = self.base_dir / key
        if not src.exists():
            raise FileNotFoundError(f"File not found: {key}")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(key).suffix)
        tmp.write(src.read_bytes())
        tmp.close()
        return Path(tmp.name)

    def upload_file(self, key: str, data: bytes) -> None:
        dest = self.base_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def get_file_info(self, key: str) -> FileItem | None:
        f = self.base_dir / key
        if not f.exists():
            return None
        stat = f.stat()
        return FileItem(
            key=key,
            filename=f.name,
            size_bytes=stat.st_size,
            last_modified=str(int(stat.st_mtime)),
            extension=f.suffix.lower(),
        )

    def delete_file(self, key: str) -> None:
        f = self.base_dir / key
        if f.exists():
            f.unlink()


class S3Storage:
    def __init__(self):
        import boto3

        kwargs = {
            "aws_access_key_id": settings.s3_access_key,
            "aws_secret_access_key": settings.s3_secret_key,
            "region_name": settings.s3_region,
        }
        if settings.s3_endpoint and not settings.s3_endpoint.startswith("https://s3.amazonaws.com"):
            kwargs["endpoint_url"] = settings.s3_endpoint

        self.client = boto3.client("s3", **kwargs)
        self.bucket = settings.s3_bucket

    def list_files(self, prefix: str) -> list[FileItem]:
        paginator = self.client.get_paginator("list_objects_v2")
        items: list[FileItem] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = Path(key).suffix.lower()
                if ext not in settings.supported_extensions:
                    continue
                items.append(
                    FileItem(
                        key=key,
                        filename=Path(key).name,
                        size_bytes=obj["Size"],
                        last_modified=obj["LastModified"].isoformat(),
                        extension=ext,
                    )
                )
        return sorted(items, key=lambda x: x.key)

    def download_to_temp(self, key: str) -> Path:
        suffix = Path(key).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self.client.download_fileobj(self.bucket, key, tmp)
        tmp.close()
        return Path(tmp.name)

    def upload_file(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_file_info(self, key: str) -> FileItem | None:
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.ClientError:
            return None
        ext = Path(key).suffix.lower()
        return FileItem(
            key=key,
            filename=Path(key).name,
            size_bytes=resp["ContentLength"],
            last_modified=resp["LastModified"].isoformat(),
            extension=ext,
        )

    def delete_file(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


@functools.lru_cache(maxsize=1)
def get_storage() -> FileStorage:
    if settings.file_backend == "s3":
        log.info("File storage backend: S3 (%s)", settings.s3_endpoint)
        return S3Storage()
    log.info("File storage backend: local (%s)", settings.data_dir)
    return LocalStorage()
