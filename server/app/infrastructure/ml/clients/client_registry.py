"""MLClientRegistry — single owner of heavy ML model lifecycle.

Responsibilities:
  - Lazy cache for expensive ML objects (embeddings, LLM, reranker, etc.)
  - Invalidation with explicit dependency relationships
  - Injectable object (replaceable with fake in tests)

Creation logic lives in ``infrastructure.ml.factories``.
The registry calls factory functions on first access and caches the result.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
import logging
import threading
from typing import TYPE_CHECKING, Self, cast

from application.ports.ml_clients import AsyncSemaphorePort, MLClientPort
from config import request_settings, settings
from domain.shared.exceptions import SemaphoreTimeoutError
from infrastructure.corpus.resources import CorpusResources
from infrastructure.ml.clients import factories
from infrastructure.ml.clients.client_lease import client_operation
from infrastructure.ml.clients.instructor_client import create_async_instructor_client
from infrastructure.ml.clients.managed_llm import ManagedInstructor, ManagedLLM
from infrastructure.utils.async_tasks import finish_owned_task


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

    from qdrant_client import QdrantClient

    from application.ports.ml_clients import EmbeddingClientPort, RerankerClientPort
    from infrastructure.budget.runtime import BudgetRuntime
    from infrastructure.corpus.snapshot import BoundQdrantClient, CorpusSnapshot
    from infrastructure.ml.clients.deepinfra_clients import DeepInfraEmbeddingsClient, DeepInfraRerankerClient
    from infrastructure.ml.clients.tei_clients import TEIEmbeddingsClient, TEIRerankerClient

log = logging.getLogger("default")


active_request_clients: ContextVar[tuple[object, dict[str, object]] | None] = ContextVar(
    "active_request_clients", default=None
)


class TimeoutSemaphore(AsyncSemaphorePort):
    """asyncio.Semaphore wrapper that raises TimeoutError if acquire exceeds timeout.

    Prevents indefinite queuing when downstream services are degraded.
    """

    def __init__(self, value: int, timeout: float) -> None:
        self._sem = asyncio.Semaphore(value)
        self._timeout = timeout

    async def acquire(self) -> bool:
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self._timeout)
        except TimeoutError as exc:
            raise SemaphoreTimeoutError from exc
        return True

    def release(self) -> None:
        self._sem.release()

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        self.release()


class MLClientRegistry(MLClientPort):
    """Process-wide cache for ML clients and infrastructure singletons.

    Lifecycle:
    - Instantiated once in ``InfrastructureContainer.init()``
    - Lazy initialization on first access (no heavy work at startup)
    - Explicit invalidation via ``invalidate_*`` methods

    Dependency graph (invalidation cascades):
        invalidate_embeddings → invalidate vector_store
        invalidate_llm → invalidate all breadth LLMs
    """

    def __init__(
        self,
        *,
        budget_runtime: BudgetRuntime | None = None,
        corpus_resources: CorpusResources | None = None,
    ) -> None:
        self.budget_runtime = budget_runtime
        self.corpus_resources = (
            corpus_resources
            if corpus_resources is not None
            else CorpusResources(factories.create_qdrant_client)
        )
        self._init_lock = threading.Lock()
        self.cached_embeddings: DeepInfraEmbeddingsClient | TEIEmbeddingsClient | None = None
        self._retired_clients: list[object] = []
        self.cached_llm: ManagedLLM | None = None
        self.cached_llm_breadth_cache: dict[str, ManagedLLM] = {}
        self.cached_fast_llm: ManagedLLM | None = None
        self.cached_reranker: DeepInfraRerankerClient | TEIRerankerClient | None = None
        self._close_task: asyncio.Task[None] | None = None
        self.cached_instructor: ManagedInstructor | None = None

        # Eager init semaphores — lightweight objects, no race conditions
        self.generation_semaphore: AsyncSemaphorePort = TimeoutSemaphore(
            settings.llm_generation_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.auxiliary_semaphore: AsyncSemaphorePort = TimeoutSemaphore(
            settings.llm_auxiliary_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.qdrant_search_semaphore: AsyncSemaphorePort = TimeoutSemaphore(
            settings.qdrant_search_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.reranker_semaphore: AsyncSemaphorePort = TimeoutSemaphore(
            settings.reranker_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.embedding_semaphore: AsyncSemaphorePort = TimeoutSemaphore(
            settings.embedding_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.ingestion_semaphore: AsyncSemaphorePort = TimeoutSemaphore(
            settings.ingestion_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.qdrant_write_semaphore: AsyncSemaphorePort = TimeoutSemaphore(
            settings.qdrant_write_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )

    def request_client(self, name: str) -> object | None:
        active = active_request_clients.get()
        return active[1].get(name) if active is not None and active[0] is self else None

    @asynccontextmanager
    async def request_operation(self) -> AsyncIterator[None]:
        """Pin settings and lease the client family across all request stages."""
        active = active_request_clients.get()

        if active is not None and active[0] is self:
            yield
            return

        # No suspension between snapshot and family capture: config publication is
        # synchronous on this event loop. Factories construct pools without I/O.
        model = settings.model_copy(deep=True)
        clients: dict[str, object] = {
            "embeddings": self.embeddings(),
            "reranker": self.reranker(),
            "llm": self.llm(),
            "fast_llm": self.fast_llm(),
            "narrow": self.llm_for_breadth("narrow"),
            "broad": self.llm_for_breadth("broad"),
            "instructor": self.instructor_client,
        }
        async with AsyncExitStack() as leases:
            for client in clients.values():
                await leases.enter_async_context(client_operation(client))

            config_token = request_settings.set(model)
            client_token = active_request_clients.set((self, clients))
            try:
                yield
            finally:
                active_request_clients.reset(client_token)
                request_settings.reset(config_token)

    # ------------------------------------------------------------------
    # Accessors (lazy init via factories, thread-safe)
    # ------------------------------------------------------------------

    def embeddings(self) -> EmbeddingClientPort:
        pinned = self.request_client("embeddings")

        if pinned is not None:
            return cast("DeepInfraEmbeddingsClient | TEIEmbeddingsClient", pinned)

        if self.cached_embeddings is None:
            with self._init_lock:
                if self.cached_embeddings is None:
                    self.cached_embeddings = factories.create_embeddings(runtime=self.budget_runtime)

        return self.cached_embeddings

    def llm(self) -> ManagedLLM:
        pinned = self.request_client("llm")

        if pinned is not None:
            return cast("ManagedLLM", pinned)

        if self.cached_llm is None:
            with self._init_lock:
                if self.cached_llm is None:
                    self.cached_llm = ManagedLLM(factories.create_llm(runtime=self.budget_runtime))

        return self.cached_llm

    def llm_for_breadth(self, breadth: str) -> ManagedLLM:
        pinned = self.request_client(breadth)

        if pinned is not None:
            return cast("ManagedLLM", pinned)

        if breadth not in self.cached_llm_breadth_cache:
            with self._init_lock:
                if breadth not in self.cached_llm_breadth_cache:
                    self.cached_llm_breadth_cache[breadth] = ManagedLLM(
                        factories.create_llm_for_breadth(breadth, runtime=self.budget_runtime)
                    )

        return self.cached_llm_breadth_cache[breadth]

    def fast_llm(self) -> ManagedLLM:
        pinned = self.request_client("fast_llm")

        if pinned is not None:
            return cast("ManagedLLM", pinned)

        """Fast, lightweight LLM for auxiliary pipeline calls (condense, relevance)."""

        if self.cached_fast_llm is None:
            with self._init_lock:
                if self.cached_fast_llm is None:
                    self.cached_fast_llm = ManagedLLM(
                        factories.create_fast_llm_for_auxiliary(runtime=self.budget_runtime)
                    )

        return self.cached_fast_llm

    def reranker(self) -> RerankerClientPort:
        pinned = self.request_client("reranker")

        if pinned is not None:
            return cast("DeepInfraRerankerClient | TEIRerankerClient", pinned)

        if self.cached_reranker is None:
            with self._init_lock:
                if self.cached_reranker is None:
                    self.cached_reranker = factories.create_reranker(runtime=self.budget_runtime)

        return self.cached_reranker

    @property
    def corpus_snapshot(self) -> CorpusSnapshot | None:
        return self.corpus_resources.corpus_snapshot

    def install_corpus(self, client: QdrantClient | BoundQdrantClient) -> None:
        return self.corpus_resources.install_corpus(client)

    def corpus_operation(self) -> AbstractAsyncContextManager[None]:
        return self.corpus_resources.corpus_operation()

    def qdrant_client(self) -> QdrantClient | BoundQdrantClient:
        return self.corpus_resources.qdrant_client()

    @property
    def instructor_client(self) -> object:
        """Cached async instructor client for structured LLM output (sufficiency, relevance)."""
        pinned = self.request_client("instructor")

        if pinned is not None:
            return cast("ManagedInstructor", pinned)

        if self.cached_instructor is None:
            with self._init_lock:
                if self.cached_instructor is None:
                    model = settings.model_copy()

                    instructor = create_async_instructor_client(
                        runtime=self.budget_runtime,
                        base_url=model.openrouter_base_url,
                        api_key=model.openrouter_api_key,
                    )

                    self.cached_instructor = ManagedInstructor(instructor)

        return self.cached_instructor

    # ------------------------------------------------------------------
    # Invalidation (with dependency cascades)
    # ------------------------------------------------------------------

    def retire_client(self, client: object) -> None:
        self._retired_clients.append(client)

        if hasattr(client, "retire"):
            client.retire()
            task = getattr(client, "_close_task", None)

            if task is not None:
                task.add_done_callback(lambda completed: self.release_retired_client(client, completed))

    def release_retired_client(self, client: object, task: asyncio.Task[None]) -> None:
        if task.cancelled() or task.exception() is not None:
            log.error("Retired ML pool cleanup failed; retained for shutdown retry")
            return

        if client in self._retired_clients:
            self._retired_clients.remove(client)

    def install_clients(self, prepared: dict[str, object]) -> None:
        retired: list[object] = []
        with self._init_lock:
            for attr, value in prepared.items():
                old = getattr(self, attr)
                retired.extend(old.values() if isinstance(old, dict) else [old])
                setattr(self, attr, value)

        for client in retired:
            if client is not None:
                self.retire_client(client)

    def invalidate_llm(self) -> None:
        """Clear cached LLM instance (model/provider/params changed).

        Cascades: clears all breadth-specific LLM caches and instructor client.
        """
        with self._init_lock:
            clients = [
                self.cached_llm,
                self.cached_fast_llm,
                self.cached_instructor,
                *self.cached_llm_breadth_cache.values(),
            ]

            for client in clients:
                if client is not None:
                    self.retire_client(client)

            self.cached_llm = None
            self.cached_llm_breadth_cache.clear()
            self.cached_fast_llm = None
            self.cached_instructor = None
        log.info("MLClientRegistry: LLM cache invalidated")

    def invalidate_embeddings(self) -> None:
        """Replace the embedding adapter on the next access."""
        with self._init_lock:
            if self.cached_embeddings is not None:
                self.retire_client(self.cached_embeddings)
                self.cached_embeddings = None
        log.info("MLClientRegistry: embedding client cache invalidated")

    def invalidate_reranker(self) -> None:
        """Replace the reranker adapter on the next access."""
        with self._init_lock:
            if self.cached_reranker is not None:
                self.retire_client(self.cached_reranker)
                self.cached_reranker = None
        log.info("MLClientRegistry: reranker client cache invalidated")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Finish owned cleanup before propagating caller cancellation."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self.close_clients())

        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await finish_owned_task(self._close_task)
            raise

    async def close_clients(self) -> None:
        """Close the owned inference and corpus pools."""
        await self.corpus_resources.drain()
        clients_to_close: list[object] = [
            self.cached_embeddings,
            self.cached_reranker,
            self.cached_instructor,
            self.cached_llm,
            self.cached_fast_llm,
        ]
        clients_to_close.extend(self._retired_clients)
        clients_to_close.extend(self.cached_llm_breadth_cache.values())

        for client in clients_to_close:
            if client is not None and hasattr(client, "close"):
                try:
                    await client.close()
                except Exception:
                    log.debug("Failed to close ML client during shutdown", exc_info=True)

        await self.corpus_resources.close()

        self.cached_embeddings = None
        self.cached_reranker = None
        self.cached_instructor = None
        self.cached_llm = None
        self.cached_fast_llm = None
        self.cached_llm_breadth_cache.clear()
        self._retired_clients.clear()
        log.info("MLClientRegistry: connection pools closed")
