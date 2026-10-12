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
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from application.ports.ml_clients import AsyncSemaphorePort, MLClientPort
from config import settings
from infrastructure.ml.clients.managed_llm import ManagedLLM, ManagedInstructor
from domain.exceptions.domain_errors import SemaphoreTimeoutError
from domain.value_objects.llm_provider import LLMProvider

from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.bm25.s3_snapshot_reader import S3SnapshotReader
from infrastructure.storage.file_storage import run_storage_io
from infrastructure.ml.clients import factories

log = logging.getLogger("default")


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
        except asyncio.TimeoutError as exc:
            raise SemaphoreTimeoutError() from exc
        return True

    def release(self) -> None:
        self._sem.release()

    async def __aenter__(self) -> "TimeoutSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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
        invalidate_bm25 → reload on next access
    """

    def __init__(self) -> None:
        self._init_lock = threading.Lock()
        self._embeddings: Any = None
        self._retired_clients: list[Any] = []
        self._llm: Any = None
        self._llm_breadth_cache: dict[str, Any] = {}
        self._fast_llm: Any = None
        self._reranker: Any = None
        self._qdrant_client: Any = None
        self._bm25_reader = S3SnapshotReader()
        self._retired_readers: list[S3SnapshotReader] = []
        self._closed = False
        self._bm25_index: BM25Index | None = None
        self._bm25_loaded: bool = False
        self._bm25_generation = 0
        self._bm25_state_lock = threading.Lock()
        self._bm25_load_lock = threading.Lock()
        self._instructor: Any = None

        # Eager init semaphores — lightweight objects, no race conditions
        self.generation_semaphore = TimeoutSemaphore(
            settings.llm_generation_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.auxiliary_semaphore = TimeoutSemaphore(
            settings.llm_auxiliary_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.qdrant_search_semaphore = TimeoutSemaphore(
            settings.qdrant_search_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.reranker_semaphore = TimeoutSemaphore(
            settings.reranker_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.bm25_search_semaphore = TimeoutSemaphore(
            settings.bm25_search_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.embedding_semaphore = TimeoutSemaphore(
            settings.embedding_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.ingestion_semaphore = TimeoutSemaphore(
            settings.ingestion_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )
        self.qdrant_write_semaphore = TimeoutSemaphore(
            settings.qdrant_write_max_concurrent, timeout=settings.semaphore_acquire_timeout
        )

    # ------------------------------------------------------------------
    # Accessors (lazy init via factories, thread-safe)
    # ------------------------------------------------------------------

    def embeddings(self) -> Any:
        if self._embeddings is None:
            with self._init_lock:
                if self._embeddings is None:
                    from infrastructure.ml.clients.factories import create_embeddings

                    self._embeddings = create_embeddings()
        return self._embeddings

    def llm(self) -> Any:
        if self._llm is None:
            with self._init_lock:
                if self._llm is None:
                    from infrastructure.ml.clients.factories import create_llm

                    self._llm = ManagedLLM(create_llm())
        return self._llm

    def llm_for_breadth(self, breadth: str) -> Any:
        if breadth not in self._llm_breadth_cache:
            with self._init_lock:
                if breadth not in self._llm_breadth_cache:
                    from infrastructure.ml.clients.factories import create_llm_for_breadth

                    self._llm_breadth_cache[breadth] = ManagedLLM(create_llm_for_breadth(breadth))
        return self._llm_breadth_cache[breadth]

    def fast_llm(self) -> Any:
        """Fast, lightweight LLM for auxiliary pipeline calls (condense, relevance)."""
        if self._fast_llm is None:
            with self._init_lock:
                if self._fast_llm is None:
                    from infrastructure.ml.clients.factories import create_fast_llm_for_auxiliary

                    self._fast_llm = ManagedLLM(create_fast_llm_for_auxiliary())
        return self._fast_llm

    def reranker(self) -> Any:
        if self._reranker is None:
            with self._init_lock:
                if self._reranker is None:
                    from infrastructure.ml.clients.factories import create_reranker

                    self._reranker = create_reranker()
        return self._reranker

    def qdrant_client(self) -> Any:
        if self._qdrant_client is None:
            with self._init_lock:
                if self._qdrant_client is None:
                    from infrastructure.ml.clients.factories import create_qdrant_client

                    self._qdrant_client = create_qdrant_client()
        return self._qdrant_client

    async def ensure_bm25_loaded(self) -> BM25Index | None:
        """Use the same single-flight loader as sync callers, outside the event loop."""
        return await run_storage_io(self.bm25_index)

    def bm25_index(self) -> BM25Index | None:
        """Blocking accessor for worker threads and synchronous consumers."""
        with self._bm25_state_lock:
            if self._closed:
                raise RuntimeError("ML registry is closed")
            if self._bm25_loaded:
                return self._bm25_index

        with self._bm25_load_lock:
            while True:
                with self._bm25_state_lock:
                    if self._bm25_loaded:
                        return self._bm25_index
                    if self._closed:
                        raise RuntimeError("ML registry is closed")
                    generation = self._bm25_generation
                    reader = self._bm25_reader
                index = factories.load_bm25_index(reader)
                with self._bm25_state_lock:
                    # Invalidation can run while S3 is loading; discard that snapshot.
                    if generation != self._bm25_generation:
                        continue
                    self._bm25_index = index
                    self._bm25_loaded = True
                    return index

    @contextmanager
    def bm25_update(self) -> Iterator[BM25Index]:
        """Keep index selection and mutation atomic with respect to invalidation."""
        while True:
            index = self.bm25_index()
            with self._bm25_state_lock:
                if not self._bm25_loaded or index is not self._bm25_index:
                    continue
                if index is None:
                    log.warning("BM25 snapshot unavailable — starting an index for incremental updates")
                    index = BM25Index([])
                    self._bm25_index = index
                yield index
                return

    @property
    def instructor_client(self):
        """Cached async instructor client for structured LLM output (sufficiency, relevance)."""
        if self._instructor is None:
            with self._init_lock:
                if self._instructor is None:
                    from infrastructure.ml.clients.instructor_client import (
                        create_async_instructor_client,
                    )

                    if settings.llm_provider == LLMProvider.OPENROUTER:
                        self._instructor = create_async_instructor_client(
                            base_url=settings.openrouter_base_url,
                            api_key=settings.openrouter_api_key,
                        )
                    else:
                        self._instructor = create_async_instructor_client(
                            base_url=f"{settings.ollama_base_url}/v1",
                            api_key="ollama",
                        )
                    self._instructor = ManagedInstructor(self._instructor)
        return self._instructor

    # ------------------------------------------------------------------
    # Invalidation (with dependency cascades)
    # ------------------------------------------------------------------

    def invalidate_llm(self) -> None:
        """Clear cached LLM instance (model/provider/params changed).

        Cascades: clears all breadth-specific LLM caches and instructor client.
        """
        with self._init_lock:
            clients = [self._llm, self._fast_llm, self._instructor, *self._llm_breadth_cache.values()]
            for client in clients:
                if client is not None:
                    client.retire()
                    self._retired_clients.append(client)
            self._llm = None
            self._llm_breadth_cache.clear()
            self._fast_llm = None
            self._instructor = None
        log.info("MLClientRegistry: LLM cache invalidated")

    def invalidate_embeddings(self) -> None:
        """Replace the embedding adapter on the next access."""
        with self._init_lock:
            if self._embeddings is not None:
                self._retired_clients.append(self._embeddings)
                self._embeddings = None
        log.info("MLClientRegistry: embedding client cache invalidated")

    def invalidate_reranker(self) -> None:
        """Replace the reranker adapter on the next access."""
        with self._init_lock:
            if self._reranker is not None:
                self._retired_clients.append(self._reranker)
                self._reranker = None
        log.info("MLClientRegistry: reranker client cache invalidated")

    def invalidate_snapshot_reader(self) -> None:
        with self._bm25_state_lock:
            self._retired_readers.append(self._bm25_reader)
            self._bm25_reader = S3SnapshotReader()
            self._bm25_generation += 1
            self._bm25_index = None
            self._bm25_loaded = False

    def invalidate_bm25(self) -> None:
        """Clear cached BM25 index (reload from disk on next access)."""
        with self._bm25_state_lock:
            self._bm25_generation += 1
            self._bm25_index = None
            self._bm25_loaded = False
        log.info("MLClientRegistry: BM25 index cache invalidated")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close_snapshot_readers(self) -> None:
        error = None
        with self._bm25_load_lock:
            with self._bm25_state_lock:
                readers = [self._bm25_reader, *self._retired_readers]
                self._retired_readers.clear()
            for reader in readers:
                try:
                    reader.close()
                except Exception as exc:
                    if error is None:
                        error = exc
        if error is not None:
            raise error

    async def close(self) -> None:
        """Drain BM25 jobs before closing their owned client pools."""
        self._closed = True
        reader_error = None
        try:
            await run_storage_io(self.close_snapshot_readers)
        except Exception as exc:
            reader_error = exc
        clients_to_close = [
            self._embeddings,
            self._reranker,
            self._instructor,
            self._llm,
            self._fast_llm,
        ]
        clients_to_close.extend(self._retired_clients)
        clients_to_close.extend(self._llm_breadth_cache.values())
        for client in clients_to_close:
            if client is not None and hasattr(client, "close"):
                try:
                    await client.close()
                except Exception:
                    log.debug("Failed to close ML client during shutdown", exc_info=True)
        if self._qdrant_client is not None and hasattr(self._qdrant_client, "close"):
            try:
                self._qdrant_client.close()
            except Exception:
                log.debug("Failed to close Qdrant client during shutdown", exc_info=True)
        self._embeddings = None
        self._reranker = None
        self._instructor = None
        self._qdrant_client = None
        self._llm = None
        self._fast_llm = None
        self._llm_breadth_cache.clear()
        self._retired_clients.clear()
        log.info("MLClientRegistry: connection pools closed")
        if reader_error is not None:
            raise reader_error
