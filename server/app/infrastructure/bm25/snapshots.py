"""Opt-in local BM25 snapshots for offline tools and persistence compatibility.

Production sparse retrieval uses Qdrant; this owner is never installed in the
ML registry or worker lifecycle.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
import threading
from typing import TYPE_CHECKING

from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.persistence import BM25SnapshotNotReadyError
from infrastructure.utils.async_tasks import finish_owned_task, run_sync_io


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator


log = logging.getLogger(__name__)
CLOSED_MESSAGE = "Local BM25 snapshots are closed"
NOT_READY_MESSAGE = "BM25 snapshot is not ready"


class LocalBM25Snapshots:
    def __init__(self, bm25_loader: Callable[[], Awaitable[BM25Index | None]]) -> None:
        self.bm25_loader = bm25_loader
        self.index: BM25Index | None = None
        self.loaded = False
        self.generation = 0
        self.state_lock = threading.Lock()
        self.load_lock = asyncio.Lock()
        self.closed = False
        self.drain_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        await self.drain()

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

                index = await self.bm25_loader()

                if await run_sync_io(self.publish_bm25_snapshot, index, generation):
                    return index

    def bm25_index(self) -> BM25Index | None:
        """Cached-only accessor: preparation belongs to async orchestration."""
        with self.state_lock:
            if self.closed:
                raise RuntimeError(CLOSED_MESSAGE)

            if not self.loaded:
                raise BM25SnapshotNotReadyError(NOT_READY_MESSAGE)

            return self.index

    @contextmanager
    def bm25_update(self) -> Iterator[BM25Index]:
        """Keep cached snapshot selection and CPU mutation atomic with invalidation."""
        with self.state_lock:
            if self.closed:
                raise RuntimeError(CLOSED_MESSAGE)

            if not self.loaded:
                raise BM25SnapshotNotReadyError(NOT_READY_MESSAGE)

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
        log.info("LocalBM25Snapshots: BM25 index cache invalidated")

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
