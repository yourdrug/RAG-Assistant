"""Lease an inference pool until its last admitted HTTP request completes."""

import asyncio
from contextlib import asynccontextmanager

import httpx

from infrastructure.storage.file_storage import finish_owned_task
from infrastructure.ml.clients.client_lease import active_client_leases


class ManagedHTTPPool:
    def __init__(self, timeout: float, limits: httpx.Limits) -> None:
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=False)
        self._active = 0
        self._retired = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._close_task: asyncio.Task | None = None

    @asynccontextmanager
    async def lease(self):
        if self._retired and id(self) not in active_client_leases.get():
            raise RuntimeError("Inference client was retired; acquire the current registry client")
        token = active_client_leases.set(active_client_leases.get() | {id(self)})
        self._active += 1
        self._idle.clear()
        try:
            yield
        finally:
            active_client_leases.reset(token)
            self._active -= 1
            if not self._active:
                self._idle.set()

    async def post(self, url: str, **kwargs) -> httpx.Response:
        async with self.lease():
            response = await self._client.post(url, **kwargs)
            response.raise_for_status()
            return response

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
        await self._client.aclose()

    async def close(self) -> None:
        self.retire()
        if self._close_task is not None:
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                await finish_owned_task(self._close_task)
                raise
