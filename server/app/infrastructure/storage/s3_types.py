"""SDK capabilities consumed by the S3 adapter, without provider-specific lifecycle types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, TypedDict


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime
    from types import TracebackType


class ListedObject(TypedDict):
    Key: str
    Size: int
    LastModified: datetime


class ObjectPage(TypedDict, total=False):
    Contents: list[ListedObject]


class ObjectInfo(TypedDict):
    ContentLength: int
    LastModified: datetime


class MultipartPart(TypedDict):
    PartNumber: int
    ETag: str


class S3Body(Protocol):
    async def read(self, amount: int) -> bytes: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, _tb: TracebackType | None
    ) -> None: ...


class DownloadResponse(TypedDict):
    Body: S3Body


class S3Paginator(Protocol):
    def paginate(self, **kwargs: object) -> AsyncIterator[ObjectPage]: ...


class S3Client(Protocol):
    def get_paginator(self, operation_name: str) -> S3Paginator: ...
    async def head_object(self, **kwargs: object) -> ObjectInfo: ...
    async def get_object(self, **kwargs: object) -> DownloadResponse: ...
    async def put_object(self, **kwargs: object) -> object: ...
    async def delete_object(self, **kwargs: object) -> object: ...
    async def copy_object(self, **kwargs: object) -> object: ...
    async def create_multipart_upload(self, **kwargs: object) -> dict[str, str]: ...
    async def abort_multipart_upload(self, **kwargs: object) -> object: ...
    async def upload_part(self, **kwargs: object) -> dict[str, str]: ...
    async def complete_multipart_upload(self, **kwargs: object) -> object: ...
