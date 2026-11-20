"""httpx transport that replays recorded LLM responses inside a benchmark scope.

Outside ``llm_replay_scope`` the transport only delegates. A real response is streamed
to the caller unchanged and recorded once its body has been read completely, so
streaming and time to first token stay real while recording.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from domain.benchmark.llm_replay import LLMReplayMode
from infrastructure.ml.replay.recording import (
    mark_replayed,
    recorded_response,
    request_key,
    request_model,
)
from infrastructure.ml.replay.session import current_replay


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from infrastructure.ml.replay.recording import RecordedResponse
    from infrastructure.ml.replay.session import ReplaySession


log = logging.getLogger("default")


class LLMReplayMissError(RuntimeError):
    """Replay-only benchmark run met an LLM request that was never recorded."""


def recordable(response: httpx.Response) -> bool:
    encoding = response.headers.get("content-encoding", "identity").lower()
    return response.is_success and encoding in {"", "identity"}


def replayed(recorded: RecordedResponse, request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        recorded.status_code,
        headers=recorded.headers,
        content=mark_replayed(recorded.body),
        request=request,
    )


def passthrough(response: httpx.Response, stream: httpx.AsyncByteStream) -> httpx.Response:
    """A fresh response over the recording stream; the caller reads the body as it arrives."""
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        stream=stream,
        extensions=response.extensions,
    )


def store_failed(session: ReplaySession, action: str, error: Exception) -> None:
    """Log a store failure by type only.

    Database errors carry the statement parameters, i.e. the recorded response, which may
    quote private documents; neither the message nor the traceback is logged.
    """
    session.store_error()
    log.warning("LLM replay: %s (%s)", action, type(error).__name__)


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
        except Exception as error:  # noqa: BLE001 — a lost recording only costs a future paid call
            store_failed(recording.session, "response not recorded", error)
        else:
            recording.session.recorded()

    async def aclose(self) -> None:
        await self._inner.aclose()


class ReplayAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        session = current_replay.get()

        if session is None or request.method != "POST":
            return await self._inner.handle_async_request(request)

        body = await request.aread()
        key = request_key(request.method, str(request.url), body, session.epoch)

        if session.mode != LLMReplayMode.REFRESH:
            try:
                recorded = await session.store.load(key)
            except Exception as error:  # noqa: BLE001 — treated as a miss below
                store_failed(session, "lookup failed", error)
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


def llm_async_http_client() -> httpx.AsyncClient:
    """HTTP client of every LLM call; replay applies only inside a benchmark scope."""
    return wrap_async_client(httpx.AsyncClient(follow_redirects=False))
