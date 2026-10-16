"""Port for the ML clients and concurrency controls used by the application.

The concrete registry owns client construction, caching, invalidation, and
shutdown. Consumers depend only on the capabilities they use, so replacing
the registry (or providing a test double) does not require importing an
infrastructure implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractContextManager
    from types import TracebackType


class EmbeddingClientPort(Protocol):
    async def embed_query(self, text: str) -> list[float]: ...
    def embed_query_sync(self, text: str) -> list[float]: ...


class LLMMessagePort(Protocol):
    @property
    def content(self) -> str | list[str | dict[str, object]]: ...


class LLMClientPort(Protocol):
    async def ainvoke(self, request: object, /) -> LLMMessagePort: ...
    def astream(self, request: object, /) -> AsyncIterator[LLMMessagePort]: ...


class RerankerClientPort(Protocol):
    async def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...
    def predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]: ...


@runtime_checkable
class AsyncSemaphorePort(Protocol):
    """Minimal async context-manager contract for bounded operations."""

    async def acquire(self) -> bool: ...

    def release(self) -> None: ...

    async def __aenter__(self) -> object: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None: ...


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

    def embeddings(self) -> EmbeddingClientPort: ...

    def llm(self) -> LLMClientPort: ...

    def llm_for_breadth(self, breadth: str) -> LLMClientPort: ...

    def fast_llm(self) -> LLMClientPort: ...

    def reranker(self) -> RerankerClientPort: ...

    # Vendor handles are opaque here; infrastructure owns their provider-specific APIs.
    def qdrant_client(self) -> object: ...

    def bm25_index(self) -> object:
        """Return only an already prepared snapshot; never load storage synchronously."""
        ...

    def bm25_update(self) -> AbstractContextManager[object]: ...

    async def ensure_bm25_loaded(self) -> object: ...

    @property
    def instructor_client(self) -> object: ...
