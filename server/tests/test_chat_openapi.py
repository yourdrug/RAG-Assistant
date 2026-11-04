"""The chat OpenAPI contract describes the SSE wire format."""

from fastapi import FastAPI

from application.rag.stream_events import MetaEvent, StatusEvent, TextChunk
from presentation.api.constants import SSE_HEARTBEAT, SSE_MEDIA_TYPE
from presentation.api.routes.chat import format_sse_event, router


def test_chat_openapi_describes_stream_and_event_payloads() -> None:
    app = FastAPI()
    app.include_router(router)
    paths = app.openapi()["paths"]
    response = paths["/chat"]["post"]["responses"]["200"]

    assert set(response["content"]) == {SSE_MEDIA_TYPE}
    content = response["content"][SSE_MEDIA_TYPE]
    assert content["schema"] == {"type": "string"}
    frames = [
        format_sse_event(event, "abc123")
        for event in (StatusEvent("retrieval"), TextChunk("Hello"), MetaEvent(1, []))
    ]
    assert all(frame is not None for frame in frames)
    assert content["example"] == SSE_HEARTBEAT + "".join(frame for frame in frames if frame is not None)

    for name in ("message", "status", "done", "error"):
        assert f"`{name}`" in response["description"]

    for field in (
        "text",
        "stage",
        "conversation_id",
        "sources",
        "confidence",
        "as_of_date",
        "request_id",
        "code",
    ):
        assert f'"{field}"' in response["description"]

    assert "application/json" in paths["/chat/sync"]["post"]["responses"]["200"]["content"]
