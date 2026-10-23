"""Owner of corpus clients, immutable snapshots, admission and shutdown."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, asynccontextmanager, contextmanager
import logging
import threading
from typing import TYPE_CHECKING

from domain.shared.exceptions import ValidationError
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.persistence import BM25SnapshotNotReady
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot, active_corpus
from infrastructure.utils.async_tasks import finish_owned_task, run_sync_io


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

    from qdrant_client import QdrantClient

log = logging.getLogger("default")
CLOSED_MESSAGE = "Corpus resources are closed"
NOT_READY_MESSAGE = "BM25 snapshot is not ready"


class CorpusResources:
    def __init__(
        self,
        qdrant_factory: Callable[[], QdrantClient],
        bm25_loader: Callable[[], Awaitable[BM25Index | None]] | None = None,
    ) -> None:
        self.qdrant_factory = qdrant_factory
        self.bm25_loader = bm25_loader
        self.corpus_readiness: Callable[[], Awaitable[bool]] | None = None
        self.client: QdrantClient | BoundQdrantClient | None = None
        self.index: BM25Index | None = None
        self.loaded = False
        self.generation = 0
        self.state_lock = threading.Lock()
        self.load_lock = asyncio.Lock()
        self.retired_corpus_clients: list[BoundQdrantClient] = []
        self.closed = False
        self.drain_task: asyncio.Task[None] | None = None
        self.close_task: asyncio.Task[None] | None = None

    def bind_readiness(self, check: Callable[[], Awaitable[bool]]) -> None:
        self.corpus_readiness = check

    @property
    def corpus_snapshot(self) -> CorpusSnapshot | None:
        pinned = active_corpus.get()
        with self.state_lock:
            client = pinned[1] if pinned is not None and pinned[0] is self else self.client
        snapshot = getattr(client, "snapshot", None)
        return snapshot if isinstance(snapshot, CorpusSnapshot) else None

    def install_corpus(self, client: QdrantClient | BoundQdrantClient, index: BM25Index | None) -> None:
        with self.state_lock:
            if self.closed:
                raise RuntimeError(CLOSED_MESSAGE)

            old = self.client
            self.client = client
            self.index = index
            self.loaded = True
            self.generation += 1

        if old is not None:
            if isinstance(old, BoundQdrantClient):
                old.close()

                if not old.closed or (old.close_task is not None and not old.close_task.done()):
                    self.retired_corpus_clients.append(old)
            else:
                old.close()

        self.retired_corpus_clients = [
            c
            for c in self.retired_corpus_clients
            if not c.closed or (c.close_task is not None and not c.close_task.done())
        ]

    @asynccontextmanager
    async def corpus_operation(self) -> AsyncIterator[None]:
        pinned = active_corpus.get()

        if pinned is not None and pinned[0] is self:
            yield
            return

        if self.corpus_readiness is not None and not await self.corpus_readiness():
            message = "Corpus generation is not ready; retry after synchronization"
            raise ValidationError(message)

        self.qdrant_client()
        with ExitStack() as leases:
            # Publication, pair selection and lease acquisition share one lock.
            with self.state_lock:
                if self.closed:
                    raise RuntimeError(CLOSED_MESSAGE)

                client = self.client
                index = self.index if self.loaded else None

                if isinstance(client, BoundQdrantClient):
                    leases.enter_context(client.lease())

            if not isinstance(client, BoundQdrantClient):
                yield
                return

            token = active_corpus.set((self, client, index))
            try:
                yield
            finally:
                active_corpus.reset(token)

    def qdrant_client(self) -> QdrantClient | BoundQdrantClient:
        pinned = active_corpus.get()

        if pinned is not None and pinned[0] is self:
            return pinned[1]

        with self.state_lock:
            if self.closed:
                raise RuntimeError(CLOSED_MESSAGE)

            if self.client is None:
                self.client = self.qdrant_factory()

            return self.client

    def bm25_state(self) -> tuple[bool, BM25Index | None, int]:
        with self.state_lock:
            return self.loaded, self.index, self.generation

    def publish_bm25_snapshot(self, index: BM25Index | None, generation: int) -> bool:
        with self.state_lock:
            if generation != self.generation:
                return False

            self.index = index
            self.loaded = True
            return True

    async def ensure_bm25_loaded(self) -> BM25Index | None:
        """One async load per generation, with CPU/state-lock work off the loop."""
        pinned = active_corpus.get()

        if pinned is not None and pinned[0] is self:
            return pinned[2]

        if self.closed:
            raise RuntimeError(CLOSED_MESSAGE)

        loaded, index, _ = await run_sync_io(self.bm25_state)

        if loaded:
            return index

        async with self.load_lock:
            while True:
                if self.closed:
                    raise RuntimeError(CLOSED_MESSAGE)

                loaded, index, generation = await run_sync_io(self.bm25_state)

                if loaded:
                    return index

                index = await self.bm25_loader() if self.bm25_loader is not None else None

                if await run_sync_io(self.publish_bm25_snapshot, index, generation):
                    return index

    def bm25_index(self) -> BM25Index | None:
        """Cached-only accessor: preparation belongs to async orchestration."""
        pinned = active_corpus.get()

        if pinned is not None and pinned[0] is self:
            return pinned[2]

        with self.state_lock:
            if self.closed:
                raise RuntimeError(CLOSED_MESSAGE)

            if not self.loaded:
                raise BM25SnapshotNotReady(NOT_READY_MESSAGE)

            return self.index

    @contextmanager
    def bm25_update(self) -> Iterator[BM25Index]:
        """Keep cached snapshot selection and CPU mutation atomic with invalidation."""
        with self.state_lock:
            if self.closed:
                raise RuntimeError(CLOSED_MESSAGE)

            if not self.loaded:
                raise BM25SnapshotNotReady(NOT_READY_MESSAGE)

            if self.index is None:
                log.warning("BM25 snapshot unavailable — starting an index for incremental updates")
                self.index = BM25Index([])

            yield self.index

    def invalidate_bm25(self) -> None:
        """Clear cached BM25 index (reload from disk on next access)."""
        with self.state_lock:
            self.generation += 1
            self.index = None
            self.loaded = False
        log.info("CorpusResources: BM25 index cache invalidated")

    async def drain(self) -> None:
        """Stop new work and wait for admitted loads and CPU mutations."""
        self.closed = True

        if self.drain_task is None:
            self.drain_task = asyncio.create_task(self.drain_jobs())

        try:
            await asyncio.shield(self.drain_task)
        except asyncio.CancelledError:
            await finish_owned_task(self.drain_task)
            raise

    async def drain_jobs(self) -> None:
        async with self.load_lock:
            await run_sync_io(self.bm25_state)

    async def close(self) -> None:
        """Complete cleanup even if the caller is cancelled repeatedly."""
        self.closed = True

        if self.close_task is None:
            self.close_task = asyncio.create_task(self.close_clients())

        try:
            await asyncio.shield(self.close_task)
        except asyncio.CancelledError:
            await finish_owned_task(self.close_task)
            raise

    async def close_clients(self) -> None:
        await self.drain()

        if self.client is not None and hasattr(self.client, "close"):
            try:
                if isinstance(self.client, BoundQdrantClient):
                    await self.client.aclose()
                else:
                    await asyncio.to_thread(self.client.close)
            except Exception:
                log.debug("Failed to close Qdrant client during shutdown", exc_info=True)

        for client in self.retired_corpus_clients:
            await client.aclose()

        self.retired_corpus_clients.clear()
        self.client = None
