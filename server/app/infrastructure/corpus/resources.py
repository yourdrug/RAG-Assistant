"""Owner of corpus clients, immutable snapshots, admission and shutdown."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, asynccontextmanager
import logging
import threading
from typing import TYPE_CHECKING

from domain.shared.exceptions import ValidationError
from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot, active_corpus
from infrastructure.utils.async_tasks import finish_owned_task


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from qdrant_client import QdrantClient

log = logging.getLogger("default")
CLOSED_MESSAGE = "Corpus resources are closed"


class CorpusResources:
    def __init__(
        self,
        qdrant_factory: Callable[[], QdrantClient | BoundQdrantClient],
    ) -> None:
        self.qdrant_factory = qdrant_factory
        self.corpus_readiness: Callable[[], Awaitable[bool]] | None = None
        self.client: QdrantClient | BoundQdrantClient | None = None
        self.state_lock = threading.Lock()
        self.retired_corpus_clients: list[BoundQdrantClient] = []
        self.closed = False
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

    def install_corpus(self, client: QdrantClient | BoundQdrantClient) -> None:
        with self.state_lock:
            if self.closed:
                raise RuntimeError(CLOSED_MESSAGE)

            old = self.client
            self.client = client

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

                if isinstance(client, BoundQdrantClient):
                    leases.enter_context(client.lease())

            if not isinstance(client, BoundQdrantClient):
                yield
                return

            token = active_corpus.set((self, client))
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

    async def drain(self) -> None:
        self.closed = True

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
