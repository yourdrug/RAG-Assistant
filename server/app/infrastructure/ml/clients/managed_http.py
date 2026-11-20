"""Lease an inference pool until its last admitted HTTP request completes."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from typing import Any, TypedDict, Unpack

import httpx

from infrastructure.ml.clients.client_lease import active_client_leases
from infrastructure.utils.async_tasks import finish_owned_task


logger = logging.getLogger(__name__)


# httpx retries only failed connection attempts (refused, TLS handshake, connect timeout), so a
# request is never sent twice; one bad upstream address no longer aborts a whole benchmark run.
CONNECT_RETRIES = 2
# A 429 means the endpoint rejected the request unprocessed, so repeating it is safe.
RATE_LIMIT_RETRIES = 2
MAX_RATE_LIMIT_WAIT_SEC = 10.0


def rate_limit_wait(response: httpx.Response, attempt: int) -> float:
    """Seconds to wait before repeating a 429: the endpoint's Retry-After, else a short backoff."""
    try:
        wait = float(response.headers.get("Retry-After", ""))
    except ValueError:
        wait = float(attempt)

    return min(max(wait, 0.0), MAX_RATE_LIMIT_WAIT_SEC)


def retrying_transport(limits: httpx.Limits) -> httpx.AsyncHTTPTransport:
    return httpx.AsyncHTTPTransport(limits=limits, retries=CONNECT_RETRIES)


def retrying_sync_transport() -> httpx.HTTPTransport:
    return httpx.HTTPTransport(retries=CONNECT_RETRIES)


class HTTPRequestOptions(TypedDict, total=False):
    json: Any
    headers: dict[str, str]


class ManagedHTTPPool:
    def __init__(self, timeout: float, limits: httpx.Limits) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout, transport=retrying_transport(limits), follow_redirects=False
        )
        self._active = 0
        self._retired = False
        self._idle = asyncio.Event()
        self._idle.set()
        self._close_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def lease(self) -> AsyncIterator[None]:
        if self._retired and id(self) not in active_client_leases.get():
            message = "Inference client was retired; acquire the current registry client"
            raise RuntimeError(message)

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

    async def post(self, url: str, **kwargs: Unpack[HTTPRequestOptions]) -> httpx.Response:
        async with self.lease():
            for attempt in range(1, RATE_LIMIT_RETRIES + 1):
                response = await self._client.post(url, **kwargs)

                if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
                    break

                wait = rate_limit_wait(response, attempt)
                logger.warning(
                    "Inference endpoint rate-limited the request; retry %d in %.1fs", attempt, wait
                )
                await asyncio.sleep(wait)
            else:
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
