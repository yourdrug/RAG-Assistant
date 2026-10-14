"""Carry an admitted operation's lease through nested inference calls."""

from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

active_client_leases: ContextVar[frozenset[int]] = ContextVar("active_client_leases", default=frozenset())


@asynccontextmanager
async def client_operation(client):
    lease = getattr(client, "lease", None)
    if lease is None:
        yield
    else:
        async with lease():
            yield


@contextmanager
def sync_client_operation(client):
    lease = getattr(client, "lease", None)
    if lease is None:
        yield
    else:
        with lease():
            yield
