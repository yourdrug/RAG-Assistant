"""Immutable physical corpus and leases for requests admitted before a switch."""

from __future__ import annotations

from typing import TYPE_CHECKING
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import asyncio
import logging
import hashlib
import threading
from inspect import signature
from config import settings


if TYPE_CHECKING:
    from collections.abc import Iterator
    from infrastructure.bm25.bm25_index import BM25Index
    from infrastructure.corpus.resources import CorpusResources


@dataclass(frozen=True)
class CorpusSnapshot:
    endpoint: str
    alias: str
    physical_collection: str
    embedding_identity: str
    generation: str = ""

    @property
    def namespace(self) -> str:
        return hashlib.sha256(
            repr((self.endpoint, self.physical_collection, self.embedding_identity, self.generation)).encode()
        ).hexdigest()


installed_corpus: CorpusSnapshot | None = None

log = logging.getLogger("default")

active_corpus: ContextVar[tuple[CorpusResources, BoundQdrantClient, BM25Index | None] | None] = ContextVar(
    'active_corpus', default=None
)


class BoundQdrantClient:
    def __init__(self, client, snapshot: CorpusSnapshot, write_guard=None):
        self.client = client
        self.snapshot = snapshot
        self.write_guard = write_guard
        self.lock = threading.Lock()
        self.active = 0
        self.retired = False
        self.closed = False
        self.close_task: asyncio.Task[None] | None = None
        self.idle = threading.Event()
        self.idle.set()

    @contextmanager
    def lease(self) -> Iterator[BoundQdrantClient]:
        with self.lock:
            if self.closed:
                raise RuntimeError('Corpus client is closed')

            self.active += 1
            self.idle.clear()
        try:
            yield self
        finally:
            with self.lock:
                self.active -= 1

                if not self.active:
                    self.idle.set()

                self.close_if_idle()

    def close_if_idle(self) -> None:
        if self.retired and not self.active and not self.closed:
            self.closed = True
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self.close_raw()
            else:
                self.close_task = loop.create_task(asyncio.to_thread(self.close_raw))

    def close_raw(self) -> None:
        try:
            self.client.close()
        except Exception:
            self.closed = False
            log.error('Corpus pool cleanup failed; retained for shutdown retry')

    async def aclose(self) -> None:
        self.close()
        await asyncio.to_thread(self.idle.wait)
        self.close()

        if self.close_task is not None:
            await self.close_task

    def close(self) -> None:
        with self.lock:
            self.retired = True
            self.close_if_idle()

    def __getattr__(self, name):
        method = getattr(self.client, name)

        if not callable(method):
            return method

        @wraps(method)
        def invoke(*args, **kwargs):
            if 'collection_name' in kwargs and not kwargs['collection_name'].endswith('__embedding_identity'):
                kwargs['collection_name'] = self.snapshot.physical_collection
            elif args and name == 'get_collection' and args[0] == self.snapshot.alias:
                args = (self.snapshot.physical_collection, *args[1:])

            with self.lease():
                if name in {
                    'upsert',
                    'delete',
                    'set_payload',
                    'overwrite_payload',
                    'delete_payload',
                    'clear_payload',
                    'create_collection',
                    'delete_collection',
                    'create_payload_index',
                    'delete_payload_index',
                    'update_collection',
                    'update_collection_aliases',
                    'upload_points',
                    'upload_collection',
                    'batch_update_points',
                }:
                    if self.write_guard is not None:
                        with self.write_guard(self.snapshot):
                            return method(*args, **kwargs)

                return method(*args, **kwargs)

        return invoke


def collection_for(client) -> str:
    snapshot = getattr(client, 'snapshot', None)

    if isinstance(snapshot, CorpusSnapshot):
        return snapshot.physical_collection

    # Compatibility for callers supplying a raw Qdrant client.
    return settings.collection_name


def pin_corpus_call(function):
    parameters = signature(function)

    @wraps(function)
    async def call(*args, **kwargs):
        values = parameters.bind(*args, **kwargs).arguments
        registry = values.get('ml_clients')

        if registry is None and 'self' in values:
            registry = getattr(values['self'], '_ml_clients', None)

        if registry is None or not callable(getattr(type(registry), 'corpus_operation', None)):
            return await function(*args, **kwargs)

        async with registry.corpus_operation():
            return await function(*args, **kwargs)

    return call
