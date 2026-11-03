from collections.abc import Coroutine
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, Literal

from infrastructure.storage.s3_types import S3Client

class AioSession:
    def create_client(
        self, service_name: Literal["s3"], **kwargs: object
    ) -> AbstractAsyncContextManager[S3Client]: ...

def get_session() -> AioSession: ...

class ClientCreatorContext:
    def __init__(self, coro: Coroutine[Any, Any, S3Client]) -> None: ...
    async def __aenter__(self) -> S3Client: ...
    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None: ...
