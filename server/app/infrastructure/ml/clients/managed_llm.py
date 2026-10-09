"""Retire LLM connection pools only after their active calls finish."""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace

from langchain_core.runnables import Runnable
from infrastructure.ml.usage_capture import (
    UsageCapturingCompletions,
    active_llm_usage,
    active_stream_response,
    record_model_response,
)

log = logging.getLogger("default")


async def close_llm_pools(client) -> None:
    """Close the owning clients exposed by OpenAI, Ollama and Instructor."""
    resources = []
    if hasattr(client, "root_async_client"):
        resources += [client.root_async_client, client.root_client]
    elif hasattr(client, "_async_client"):
        resources += [client._async_client, client._client]
    elif hasattr(client, "client"):  # Instructor owns an AsyncOpenAI client.
        resources.append(client.client)
    else:
        resources.append(client)
    seen = set()
    for resource in resources:
        # Ollama's wrapper owns an httpx client.
        if type(resource).__module__.startswith("ollama"):
            resource = resource._client
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if close is not None:
            if inspect.iscoroutinefunction(close):
                await close()
            else:
                result = await asyncio.to_thread(close)
                if inspect.isawaitable(result):
                    await result


class ManagedLLM(Runnable):
    """Runnable adapter with leases for generation, auxiliary and summary calls."""

    def __init__(self, client):
        self.client = client
        if hasattr(client, 'root_async_client') and hasattr(client, 'async_client'):
            if not isinstance(client.async_client, UsageCapturingCompletions):
                client.async_client = UsageCapturingCompletions(client.async_client)
        self._active = 0
        self._retired = False
        self._close_task = None
        self._idle = asyncio.Event()
        self._idle.set()

    def __getattr__(self, name):
        return getattr(self.client, name)

    @asynccontextmanager
    async def _lease(self):
        if self._retired:
            raise RuntimeError("LLM client was retired; acquire the current client from the registry")
        self._active += 1
        self._idle.clear()
        try:
            yield
        finally:
            self._active -= 1
            if self._active == 0:
                self._idle.set()

    def invoke(self, input, config=None, **kwargs):
        raise RuntimeError("Managed LLM clients require async invocation")

    async def ainvoke(self, input, config=None, **kwargs):
        async with self._lease():
            response = await self.client.ainvoke(input, config=config, **kwargs)
            record_model_response(
                response, model=getattr(self.client, 'model_name', '') or getattr(self.client, 'model', '')
            )
            return response

    async def astream(self, input, config=None, **kwargs):
        async with self._lease():
            accumulated = None
            metadata: dict = {}
            raw_response: dict = {}
            capture_token = active_stream_response.set(
                raw_response if active_llm_usage.get() is not None else None
            )
            try:
                async for item in self.client.astream(input, config=config, **kwargs):
                    if active_llm_usage.get() is not None:
                        metadata.update(getattr(item, 'response_metadata', {}) or {})
                        accumulated = item if accumulated is None else accumulated + item
                    yield item
            finally:
                if accumulated is not None:
                    # LangChain concatenates repeated string metadata when merging chunks.
                    accumulated.response_metadata = metadata
                    record_model_response(
                        SimpleNamespace(**raw_response) if raw_response.get('usage') else accumulated,
                        model=getattr(self.client, 'model_name', '') or getattr(self.client, 'model', ''),
                    )
                active_stream_response.reset(capture_token)

    def retire(self) -> None:
        self._retired = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # Shutdown will close it when invalidation came from a sync caller.
        if self._close_task is None:
            self._close_task = loop.create_task(self._close_when_idle())

    async def _close_when_idle(self):
        await self._idle.wait()
        try:
            await close_llm_pools(self.client)
        except Exception:
            log.exception("Failed to close retired LLM pools")
            raise

    async def close(self):
        self.retire()
        if self._close_task is not None:
            await self._close_task


class ManagedInstructor(ManagedLLM):
    """Expose Instructor's async completion call through the same lease."""

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        async with self._lease():
            return await self.client.chat.completions.create(**kwargs)
