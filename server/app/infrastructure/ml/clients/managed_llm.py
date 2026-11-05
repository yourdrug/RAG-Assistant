"""Retire LLM connection pools only after their active calls finish."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, Unpack, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from infrastructure.ml.clients.client_lease import active_client_leases
from infrastructure.ml.usage_capture import (
    UsageCapturingCompletions,
    active_llm_usage,
    active_stream_response,
    record_model_response,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from instructor import AsyncInstructor
    from langchain_core.messages import BaseMessageChunk

    from infrastructure.ml.clients.request_types import CompletionRequest

log = logging.getLogger("default")


def llm_pool_resources(client: object) -> list[object]:
    for names in [("root_async_client", "root_client"), ("_async_client", "_client"), ("client",)]:
        if hasattr(client, names[0]):
            return [getattr(client, name) for name in names]

    return [client]


async def close_llm_pools(client: object) -> None:
    """Close the owning clients exposed by OpenAI and Instructor."""
    seen: set[int] = set()

    for candidate in llm_pool_resources(client):
        resource = candidate

        if resource is None or id(resource) in seen:
            continue

        seen.add(id(resource))
        close = cast(
            "Callable[[], object] | None",
            getattr(resource, "aclose", None) or getattr(resource, "close", None),
        )

        if close is not None:
            if inspect.iscoroutinefunction(close):
                await close()
            else:
                result = await asyncio.to_thread(close)

                if inspect.isawaitable(result):
                    await result


class ProviderRunnable(Protocol):
    async def ainvoke(self, *args: object, **kwargs: object) -> BaseMessage: ...
    def astream(self, *args: object, **kwargs: object) -> AsyncIterator[BaseMessage]: ...


class ManagedLLM(Runnable[object, BaseMessage]):
    """Runnable adapter with leases for generation, auxiliary and summary calls."""

    def __init__(self, client: object) -> None:
        self.client = client

        if (
            hasattr(client, "root_async_client")
            and hasattr(client, "async_client")
            and not isinstance(client.async_client, UsageCapturingCompletions)
        ):
            client.async_client = UsageCapturingCompletions(client.async_client)

        self._active = 0
        self._retired = False
        self._close_task: asyncio.Task[None] | None = None
        self._idle = asyncio.Event()
        self._idle.set()

    def __getattr__(self, name: str) -> object:
        return getattr(self.client, name)

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[None]:
        if self._retired and id(self) not in active_client_leases.get():
            message = "LLM client was retired; acquire the current client from the registry"
            raise RuntimeError(message)

        token = active_client_leases.set(active_client_leases.get() | {id(self)})
        self._active += 1
        self._idle.clear()
        try:
            yield
        finally:
            active_client_leases.reset(token)
            self._active -= 1

            if self._active == 0:
                self._idle.set()

    def invoke(self, *args: object, **kwargs: object) -> BaseMessage:
        del args, kwargs
        message = "Managed LLM clients require async invocation"
        raise RuntimeError(message)

    async def ainvoke(self, *args: object, **kwargs: object) -> BaseMessage:
        async with self.lease():
            response = await cast("ProviderRunnable", self.client).ainvoke(*args, **kwargs)
            record_model_response(
                response, model=getattr(self.client, "model_name", "") or getattr(self.client, "model", "")
            )
            return response

    async def astream(self, *args: object, **kwargs: object) -> AsyncIterator[BaseMessage]:
        async with self.lease():
            accumulated: BaseMessageChunk | None = None
            metadata: dict[str, object] = {}
            raw_response: dict[str, object] = {}
            capture_token = active_stream_response.set(
                raw_response if active_llm_usage.get() is not None else None
            )
            try:
                async for item in cast("ProviderRunnable", self.client).astream(*args, **kwargs):
                    if active_llm_usage.get() is not None:
                        metadata.update(getattr(item, "response_metadata", {}) or {})
                        chunk = cast("BaseMessageChunk", item)
                        accumulated = chunk if accumulated is None else accumulated + chunk

                    yield item
            finally:
                if accumulated is not None:
                    accumulated.response_metadata = metadata
                    record_model_response(
                        SimpleNamespace(**raw_response) if raw_response.get("usage") else accumulated,
                        model=getattr(self.client, "model_name", "") or getattr(self.client, "model", ""),
                    )

                active_stream_response.reset(capture_token)

    def retire(self) -> None:
        self._retired = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        if self._close_task is None:
            self._close_task = loop.create_task(self.close_when_idle())

    async def close_when_idle(self) -> None:
        await self._idle.wait()
        try:
            await close_llm_pools(self.client)
        except Exception:
            log.exception("Failed to close retired LLM pools")
            raise

    async def close(self) -> None:
        self.retire()

        if self._close_task is not None:
            await self._close_task


class ManagedInstructor(ManagedLLM):
    """Expose Instructor's async completion call through the same lease."""

    @property
    def chat(self) -> SimpleNamespace:
        return SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create[T: BaseModel](self, **kwargs: Unpack[CompletionRequest[T]]) -> T:
        async with self.lease():
            return await cast("AsyncInstructor", self.client).chat.completions.create(**kwargs)
