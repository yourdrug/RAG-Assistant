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
from typing import TYPE_CHECKING, Any

from config import settings
from domain.value_objects.llm_provider import LLMProvider

if TYPE_CHECKING:
    from infrastructure.bm25.hybrid import BM25Index

log = logging.getLogger("default")


class MLClientRegistry:
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
        self._embeddings: Any = None
        self._llm: Any = None
        self._llm_breadth_cache: dict[str, Any] = {}
        self._fast_llm: Any = None
        self._reranker: Any = None
        self._qdrant_client: Any = None
        self._bm25_index: BM25Index | None = None
        self._bm25_loaded: bool = False
        self._generation_semaphore: asyncio.Semaphore | None = None
        self._auxiliary_semaphore: asyncio.Semaphore | None = None
        self._qdrant_search_semaphore: asyncio.Semaphore | None = None
        self._instructor: Any = None

    # ------------------------------------------------------------------
    # Accessors (lazy init via factories)
    # ------------------------------------------------------------------

    def embeddings(self) -> Any:
        if self._embeddings is None:
            from infrastructure.ml.clients.factories import create_embeddings

            self._embeddings = create_embeddings()
        return self._embeddings

    def llm(self) -> Any:
        if self._llm is None:
            from infrastructure.ml.clients.factories import create_llm

            self._llm = create_llm()
        return self._llm

    def llm_for_breadth(self, breadth: str) -> Any:
        if breadth not in self._llm_breadth_cache:
            from infrastructure.ml.clients.factories import create_llm_for_breadth

            self._llm_breadth_cache[breadth] = create_llm_for_breadth(breadth)
        return self._llm_breadth_cache[breadth]

    def fast_llm(self) -> Any:
        """Fast, lightweight LLM for auxiliary pipeline calls (condense, relevance)."""
        if self._fast_llm is None:
            from infrastructure.ml.clients.factories import create_fast_llm_for_auxiliary

            self._fast_llm = create_fast_llm_for_auxiliary()
        return self._fast_llm

    def reranker(self) -> Any:
        if self._reranker is None:
            from infrastructure.ml.clients.factories import create_reranker

            self._reranker = create_reranker()
        return self._reranker

    def qdrant_client(self) -> Any:
        if self._qdrant_client is None:
            from infrastructure.ml.clients.factories import create_qdrant_client

            self._qdrant_client = create_qdrant_client()
        return self._qdrant_client

    def bm25_index(self) -> BM25Index | None:
        if not self._bm25_loaded:
            from infrastructure.ml.clients.factories import load_bm25_index

            self._bm25_index = load_bm25_index()
            self._bm25_loaded = True
        return self._bm25_index

    @property
    def generation_semaphore(self) -> asyncio.Semaphore:
        """Semaphore for heavy LLM operations (generation, 10-30s)."""
        if self._generation_semaphore is None:
            self._generation_semaphore = asyncio.Semaphore(settings.llm_generation_max_concurrent)
        return self._generation_semaphore

    @property
    def auxiliary_semaphore(self) -> asyncio.Semaphore:
        """Semaphore for light LLM operations (condense, relevance, 1-3s)."""
        if self._auxiliary_semaphore is None:
            self._auxiliary_semaphore = asyncio.Semaphore(settings.llm_auxiliary_max_concurrent)
        return self._auxiliary_semaphore

    @property
    def qdrant_search_semaphore(self) -> asyncio.Semaphore:
        """Semaphore for Qdrant search operations (prevents thundering herd on decomposition retries)."""
        if self._qdrant_search_semaphore is None:
            self._qdrant_search_semaphore = asyncio.Semaphore(settings.qdrant_search_max_concurrent)
        return self._qdrant_search_semaphore

    @property
    def instructor_client(self):
        """Cached async instructor client for structured LLM output (sufficiency, relevance)."""
        if self._instructor is None:
            from infrastructure.ml.clients.instructor_client import create_async_instructor_client

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
        return self._instructor

    # ------------------------------------------------------------------
    # Invalidation (with dependency cascades)
    # ------------------------------------------------------------------

    def invalidate_llm(self) -> None:
        """Clear cached LLM instance (model/provider/params changed).

        Cascades: clears all breadth-specific LLM caches and instructor client.
        """
        self._llm = None
        self._llm_breadth_cache.clear()
        self._fast_llm = None
        self._instructor = None
        log.info("MLClientRegistry: LLM cache invalidated")

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
        for client in (self._embeddings, self._reranker, self._instructor):
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
        log.info("MLClientRegistry: connection pools closed")
