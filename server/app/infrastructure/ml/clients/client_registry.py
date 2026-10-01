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
import time
from typing import TYPE_CHECKING, Any

from application.ports.ml_clients import AsyncSemaphorePort, MLClientPort
from config import settings
from infrastructure.ml.clients.managed_llm import ManagedLLM, ManagedInstructor
from domain.exceptions.domain_errors import SemaphoreTimeoutError
from domain.value_objects.llm_provider import LLMProvider

if TYPE_CHECKING:
    from infrastructure.bm25.bm25_index import BM25Index

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
        self._bm25_index: BM25Index | None = None
        self._bm25_loaded: bool = False
        self._bm25_last_reload: float = 0.0
        self._bm25_reload_lock: asyncio.Lock | None = None
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

    async def _ensure_bm25_loaded(self) -> BM25Index | None:
        """Load BM25 index from S3 with thundering-herd protection.

        Uses an asyncio.Lock so that concurrent coroutines after invalidation
        do not all trigger a synchronous S3 download simultaneously.  The
        threading-level double-checked lock (``_init_lock``) is kept for
        thread-safety during the very first access from a non-async context.
        """
        if self._bm25_loaded:
            return self._bm25_index

        if self._bm25_reload_lock is None:
            self._bm25_reload_lock = asyncio.Lock()

        async with self._bm25_reload_lock:
            # Re-check after acquiring lock — another coroutine may have reloaded.
            now = time.monotonic()
            if self._bm25_loaded or (now - self._bm25_last_reload) <= 5.0:
                return self._bm25_index

            from infrastructure.ml.clients.factories import load_bm25_index

            self._bm25_index = await asyncio.to_thread(load_bm25_index)
            self._bm25_loaded = True
            self._bm25_last_reload = now
            return self._bm25_index

    def bm25_index(self) -> BM25Index | None:
        """Return the BM25 index, blocking on first load (thread-safe via _init_lock)."""
        if not self._bm25_loaded:
            with self._init_lock:
                now = time.monotonic()
                if not self._bm25_loaded and (now - self._bm25_last_reload) > 5.0:
                    from infrastructure.ml.clients.factories import load_bm25_index

                    self._bm25_index = load_bm25_index()
                    self._bm25_loaded = True
                    self._bm25_last_reload = now
        return self._bm25_index

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

    def invalidate_bm25(self) -> None:
        """Clear cached BM25 index (reload from disk on next access)."""
        self._bm25_index = None
        self._bm25_loaded = False
        log.info("MLClientRegistry: BM25 index cache invalidated")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all HTTP connection pools. Call during app shutdown."""
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
