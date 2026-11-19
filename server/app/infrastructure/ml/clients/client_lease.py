"""Carry an admitted operation's lease through nested inference calls."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, cast


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager


active_client_leases: ContextVar[frozenset[int]] = ContextVar("active_client_leases", default=frozenset())


@asynccontextmanager
async def client_operation(client: object) -> AsyncIterator[None]:
    lease = getattr(client, "lease", None)

    if lease is None:
        yield
    else:
        async with cast("Callable[[], AbstractAsyncContextManager[object]]", lease)():
            yield
