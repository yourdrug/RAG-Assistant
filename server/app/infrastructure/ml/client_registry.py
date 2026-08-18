"""MLClientRegistry — single owner of heavy ML model lifecycle.

Created once in ``build_container()``, injected into services via constructor.
Invalidation is explicit via methods, not via ``module.cache_clear()`` from
an unrelated module.

The registry delegates to the existing ``infrastructure.clients`` factory
functions internally.  The difference: the registry is an **injectable
object** (can be replaced with a fake in tests), whereas bare
``get_llm()`` calls are hidden dependencies that require monkeypatching.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("default")


class MLClientRegistry:
    """Process-wide registry for ML clients and infrastructure singletons.

    Lifecycle:
    - Instantiated once in ``build_container()``
    - Lazy initialization on first access (no heavy work at startup)
    - Explicit invalidation via ``invalidate_*`` methods
    """

    def __init__(self) -> None:
        self._embeddings: Any = None
        self._vector_store: Any = None
        self._llm: Any = None
        self._llm_breadth_cache: dict[str, Any] = {}
        self._reranker: Any = None
        self._qdrant_client: Any = None
        self._bm25_index: Any = None
        self._bm25_loaded: bool = False

    # ------------------------------------------------------------------
    # Accessors (lazy init)
    # ------------------------------------------------------------------

    def embeddings(self):
        if self._embeddings is None:
            from infrastructure.clients import get_embeddings

            self._embeddings = get_embeddings()
        return self._embeddings

    def vector_store(self):
        if self._vector_store is None:
            from infrastructure.clients import get_vector_store

            self._vector_store = get_vector_store()
        return self._vector_store

    def llm(self):
        if self._llm is None:
            from infrastructure.clients import get_llm

            self._llm = get_llm()
        return self._llm

    def llm_for_breadth(self, breadth: str):
        if breadth not in self._llm_breadth_cache:
            from infrastructure.clients import get_llm_for_breadth

            self._llm_breadth_cache[breadth] = get_llm_for_breadth(breadth)
        return self._llm_breadth_cache[breadth]

    def reranker(self):
        if self._reranker is None:
            from infrastructure.clients import get_reranker

            self._reranker = get_reranker()
        return self._reranker

    def qdrant_client(self):
        if self._qdrant_client is None:
            from infrastructure.clients import get_qdrant_client

            self._qdrant_client = get_qdrant_client()
        return self._qdrant_client

    def bm25_index(self):
        if not self._bm25_loaded:
            from infrastructure.clients import get_bm25_index

            self._bm25_index = get_bm25_index()
            self._bm25_loaded = True
        return self._bm25_index

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_llm(self) -> None:
        """Clear cached LLM instance (model/provider/params changed)."""
        self._llm = None
        self._llm_breadth_cache.clear()
        log.info("MLClientRegistry: LLM cache invalidated")

    def invalidate_embeddings(self) -> None:
        """Clear cached embeddings model."""
        self._embeddings = None
        self._vector_store = None
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
