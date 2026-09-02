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
from typing import Any

from config import settings

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
        self._reranker: Any = None
        self._qdrant_client: Any = None
        self._bm25_index: Any = None
        self._bm25_loaded: bool = False
        self._llm_semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # Accessors (lazy init via factories)
    # ------------------------------------------------------------------

    def embeddings(self):
        if self._embeddings is None:
            from infrastructure.ml.factories import create_embeddings

            self._embeddings = create_embeddings()
        return self._embeddings

    def llm(self):
        if self._llm is None:
            from infrastructure.ml.factories import create_llm

            self._llm = create_llm()
        return self._llm

    def llm_for_breadth(self, breadth: str):
        if breadth not in self._llm_breadth_cache:
            from infrastructure.ml.factories import create_llm_for_breadth

            self._llm_breadth_cache[breadth] = create_llm_for_breadth(breadth)
        return self._llm_breadth_cache[breadth]

    def reranker(self):
        if self._reranker is None:
            from infrastructure.ml.factories import create_reranker

            self._reranker = create_reranker()
        return self._reranker

    def qdrant_client(self):
        if self._qdrant_client is None:
            from infrastructure.ml.factories import create_qdrant_client

            self._qdrant_client = create_qdrant_client()
        return self._qdrant_client

    def bm25_index(self):
        if not self._bm25_loaded:
            from infrastructure.ml.factories import load_bm25_index

            self._bm25_index = load_bm25_index()
            self._bm25_loaded = True
        return self._bm25_index

    @property
    def llm_semaphore(self) -> asyncio.Semaphore:
        if self._llm_semaphore is None:
            self._llm_semaphore = asyncio.Semaphore(settings.llm_max_concurrent)
        return self._llm_semaphore

    # ------------------------------------------------------------------
    # Invalidation (with dependency cascades)
    # ------------------------------------------------------------------

    def invalidate_llm(self) -> None:
        """Clear cached LLM instance (model/provider/params changed).

        Cascades: clears all breadth-specific LLM caches too.
        """
        self._llm = None
        self._llm_breadth_cache.clear()
        log.info("MLClientRegistry: LLM cache invalidated")

    def invalidate_embeddings(self) -> None:
        """Clear cached embeddings model."""
        self._embeddings = None
        log.info("MLClientRegistry: embeddings cache invalidated")

    def invalidate_bm25(self) -> None:
        """Clear cached BM25 index (reload from disk on next access)."""
        self._bm25_index = None
        self._bm25_loaded = False
        log.info("MLClientRegistry: BM25 index cache invalidated")

    def invalidate_reranker(self) -> None:
        """Clear cached reranker model."""
        self._reranker = None
        log.info("MLClientRegistry: reranker cache invalidated")

    def invalidate_qdrant(self) -> None:
        """Clear cached Qdrant client."""
        self._qdrant_client = None
        log.info("MLClientRegistry: Qdrant client cache invalidated")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close all HTTP connection pools. Call during app shutdown."""
        for client in (self._embeddings, self._reranker):
            if client is not None and hasattr(client, "close"):
                try:
                    await client.close()
                except Exception:
                    pass
        self._embeddings = None
        self._reranker = None
        log.info("MLClientRegistry: connection pools closed")
