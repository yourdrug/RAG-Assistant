"""FastAPI server spans with the request's paid model-call tally and request id."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from infrastructure.telemetry.tracing import annotate_current_span, paid_call_scope, suppress_instrumentation


if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Probes and scrapes would flood the trace store without showing any request path.
EXCLUDED_URLS = "/health$,/live$,/ready$,/metrics$"
EXCLUDED_PATH = re.compile("|".join(EXCLUDED_URLS.split(",")))


class PaidCallTallyMiddleware:
    """Pure ASGI, so the tally also covers streamed (SSE) bodies; runs inside the server span."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if EXCLUDED_PATH.search(scope.get("path", "")):
            # Their Qdrant/LLM probes would otherwise surface as parentless client traces.
            with suppress_instrumentation():
                await self.app(scope, receive, send)
            return

        with paid_call_scope() as tally:

            async def send_annotated(message: Message) -> None:
                if message["type"] == "http.response.start":
                    for name, value in message.get("headers", ()):
                        if name.lower() == b"x-request-id":
                            annotate_current_span(**{"rag.request_id": value.decode("latin-1")})
                elif message["type"] == "http.response.body" and not message.get("more_body", False):
                    # The server span ends with the last body message, before this middleware returns.
                    annotate_current_span(**tally.attributes())

                await send(message)

            await self.app(scope, receive, send_annotated)


def instrument_api(app: FastAPI) -> None:
    """Call after the other middlewares: the tally must wrap them, the server span wraps it."""
    app.add_middleware(PaidCallTallyMiddleware)
    FastAPIInstrumentor.instrument_app(app, excluded_urls=EXCLUDED_URLS)
