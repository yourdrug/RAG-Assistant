"""Port for the ML clients and concurrency controls used by the application.

The concrete registry owns client construction, caching, invalidation, and
shutdown. Consumers depend only on the capabilities they use, so replacing
the registry (or providing a test double) does not require importing an
infrastructure implementation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AsyncSemaphorePort(Protocol):
    """Minimal async context-manager contract for bounded operations."""

    async def acquire(self) -> bool: ...

    def release(self) -> None: ...

    async def __aenter__(self) -> Any: ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...


@runtime_checkable
class MLClientPort(Protocol):
    """Capabilities consumed from the process-wide ML client provider.

    This is deliberately a port rather than an alias for
    ``MLClientRegistry``. The registry is an infrastructure lifecycle
    component; this contract describes only what RAG, summaries, BM25, and
    vector storage need from it.
    """

    generation_semaphore: AsyncSemaphorePort
    auxiliary_semaphore: AsyncSemaphorePort
    qdrant_search_semaphore: AsyncSemaphorePort
    reranker_semaphore: AsyncSemaphorePort
    bm25_search_semaphore: AsyncSemaphorePort
    embedding_semaphore: AsyncSemaphorePort
    ingestion_semaphore: AsyncSemaphorePort
    qdrant_write_semaphore: AsyncSemaphorePort

    def embeddings(self) -> Any: ...

    def llm(self) -> Any: ...

    def llm_for_breadth(self, breadth: str) -> Any: ...

    def fast_llm(self) -> Any: ...

    def reranker(self) -> Any: ...

    def qdrant_client(self) -> Any: ...

    def bm25_index(self) -> Any: ...

    async def _ensure_bm25_loaded(self) -> Any: ...

    @property
    def instructor_client(self) -> Any: ...
