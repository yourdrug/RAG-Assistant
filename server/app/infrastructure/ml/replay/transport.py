"""httpx transports that replay recorded LLM responses inside a benchmark scope.

Outside ``llm_replay_scope`` both transports only delegate. A real response is streamed
to the caller unchanged and recorded once its body has been read completely, so
streaming and time to first token stay real while recording.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx

from domain.benchmark.llm_replay import LLMReplayMode
from infrastructure.ml.replay.recording import recorded_response, request_key, request_model
from infrastructure.ml.replay.session import current_replay


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine, Iterator

    from infrastructure.ml.replay.recording import RecordedResponse
    from infrastructure.ml.replay.session import ReplaySession


log = logging.getLogger("default")

STORE_TIMEOUT = 30.0


class LLMReplayMissError(RuntimeError):
    """Replay-only benchmark run met an LLM request that was never recorded."""


def recordable(response: httpx.Response) -> bool:
    encoding = response.headers.get("content-encoding", "identity").lower()
    return response.is_success and encoding in {"", "identity"}


def replayed(recorded: RecordedResponse, request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        recorded.status_code, headers=recorded.headers, content=recorded.body, request=request
    )


def passthrough(
    response: httpx.Response, stream: httpx.AsyncByteStream | httpx.SyncByteStream
) -> httpx.Response:
    """A fresh response over the recording stream; the caller reads the body as it arrives."""
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        stream=stream,
        extensions=response.extensions,
    )


def miss_error(request: httpx.Request) -> LLMReplayMissError:
    return LLMReplayMissError(f"No recorded LLM response for {request.url.path}; replay-only run")


class Recording:
    """Request identity and the response collected for one recorded call."""

    def __init__(self, session: ReplaySession, key: str, model: str | None) -> None:
        self.session = session
        self.key = key
        self.model = model

    def response(self, response: httpx.Response, body: bytes) -> RecordedResponse:
        return recorded_response(response.status_code, dict(response.headers), body)


class RecordingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, inner: httpx.AsyncByteStream, response: httpx.Response, recording: Recording) -> None:
        self._inner = inner
        self._response = response
        self._recording = recording

    async def __aiter__(self) -> AsyncIterator[bytes]:
        chunks: list[bytes] = []

        async for chunk in self._inner:
            chunks.append(chunk)
            yield chunk

        recording = self._recording
        try:
            await recording.session.store.save(
                recording.key, recording.model, recording.response(self._response, b"".join(chunks))
            )
        except Exception:  # a lost recording only costs a future paid call
            recording.session.store_error()
            log.warning("LLM replay: response not recorded", exc_info=True)
        else:
            recording.session.recorded()

    async def aclose(self) -> None:
        await self._inner.aclose()


class RecordingSyncStream(httpx.SyncByteStream):
    def __init__(self, inner: httpx.SyncByteStream, response: httpx.Response, recording: Recording) -> None:
        self._inner = inner
        self._response = response
        self._recording = recording

    def __iter__(self) -> Iterator[bytes]:
        chunks: list[bytes] = []

        for chunk in self._inner:
            chunks.append(chunk)
            yield chunk

        recording = self._recording
        try:
            run_on_loop(
                recording.session,
                recording.session.store.save(
                    recording.key, recording.model, recording.response(self._response, b"".join(chunks))
                ),
            )
        except Exception:  # a lost recording only costs a future paid call
            recording.session.store_error()
            log.warning("LLM replay: response not recorded", exc_info=True)
        else:
            recording.session.recorded()

    def close(self) -> None:
        self._inner.close()


def run_on_loop[T](session: ReplaySession, coroutine: Coroutine[object, object, T]) -> T:
    """Run a store call from a worker thread on the loop that owns the database pool."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run_coroutine_threadsafe(coroutine, session.loop).result(timeout=STORE_TIMEOUT)

    coroutine.close()
    message = "Synchronous LLM client called on the event loop thread"
    raise RuntimeError(message)


class ReplayAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        session = current_replay.get()

        if session is None or request.method != "POST":
            return await self._inner.handle_async_request(request)

        body = await request.aread()
        key = request_key(request.method, str(request.url), body)

        if session.mode != LLMReplayMode.REFRESH:
            try:
                recorded = await session.store.load(key)
            except Exception:  # treated as a miss below
                session.store_error()
                log.warning("LLM replay: lookup failed", exc_info=True)
                recorded = None

            if recorded is not None:
                session.hit(recorded)
                return replayed(recorded, request)

            session.miss()

            if session.mode == LLMReplayMode.REPLAY_ONLY:
                raise miss_error(request)

        # A plain body can be recorded and replayed as the client parsed it.
        request.headers["accept-encoding"] = "identity"
        response = await self._inner.handle_async_request(request)

        if not recordable(response) or not isinstance(response.stream, httpx.AsyncByteStream):
            return response

        recording = Recording(session, key, request_model(body))
        return passthrough(response, RecordingAsyncStream(response.stream, response, recording))

    async def aclose(self) -> None:
        await self._inner.aclose()


class ReplaySyncTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        session = current_replay.get()

        if session is None or request.method != "POST":
            return self._inner.handle_request(request)

        body = request.read()
        key = request_key(request.method, str(request.url), body)

        if session.mode != LLMReplayMode.REFRESH:
            try:
                recorded = run_on_loop(session, session.store.load(key))
            except Exception:  # treated as a miss below
                session.store_error()
                log.warning("LLM replay: lookup failed", exc_info=True)
                recorded = None

            if recorded is not None:
                session.hit(recorded)
                return replayed(recorded, request)

            session.miss()

            if session.mode == LLMReplayMode.REPLAY_ONLY:
                raise miss_error(request)

        request.headers["accept-encoding"] = "identity"
        response = self._inner.handle_request(request)

        if not recordable(response) or not isinstance(response.stream, httpx.SyncByteStream):
            return response

        recording = Recording(session, key, request_model(body))
        return passthrough(response, RecordingSyncStream(response.stream, response, recording))

    def close(self) -> None:
        self._inner.close()


def wrap_async_client(client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Wrap every transport the client already chose, proxy mounts from the environment included.

    Passing ``transport=`` to httpx would disable environment proxies, so the client is built
    as before and its transports are wrapped afterwards.
    """
    client._transport = ReplayAsyncTransport(client._transport)  # noqa: SLF001
    client._mounts = {  # noqa: SLF001
        pattern: None if transport is None else ReplayAsyncTransport(transport)
        for pattern, transport in client._mounts.items()  # noqa: SLF001
    }
    return client


def wrap_sync_client(client: httpx.Client) -> httpx.Client:
    client._transport = ReplaySyncTransport(client._transport)  # noqa: SLF001
    client._mounts = {  # noqa: SLF001
        pattern: None if transport is None else ReplaySyncTransport(transport)
        for pattern, transport in client._mounts.items()  # noqa: SLF001
    }
    return client


def llm_async_http_client() -> httpx.AsyncClient:
    """HTTP client of every LLM call; replay applies only inside a benchmark scope."""
    return wrap_async_client(httpx.AsyncClient(follow_redirects=False))


def llm_sync_http_client() -> httpx.Client:
    return wrap_sync_client(httpx.Client(follow_redirects=False))
