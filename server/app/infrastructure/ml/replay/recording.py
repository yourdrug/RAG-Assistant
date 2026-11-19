"""Request identity and recorded response format of the LLM replay cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


# Bump to invalidate every recorded response, e.g. after a provider silently changed a model.
KEY_VERSION = "v1"

# Response headers needed to replay the body as the client originally parsed it.
REPLAYED_HEADERS = ("content-type",)


@dataclass(frozen=True)
class RecordedResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes
    input_tokens: int | None = None
    output_tokens: int | None = None


def canonical_body(body: bytes) -> bytes:
    """JSON bodies are compared by content, not by key order or whitespace."""
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return body

    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def request_key(method: str, url: str, body: bytes) -> str:
    """Identity of an LLM request. Headers, and so the API key, never take part."""
    digest = hashlib.sha256()

    for part in (KEY_VERSION.encode(), method.upper().encode(), url.encode(), canonical_body(body)):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)

    return digest.hexdigest()


def request_model(body: bytes) -> str | None:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return None

    model = payload.get("model") if isinstance(payload, dict) else None
    return model[:255] if isinstance(model, str) else None


def usage_tokens(usage: object) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None

    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    return (
        prompt if isinstance(prompt, int) else None,
        completion if isinstance(completion, int) else None,
    )


def response_usage(body: bytes) -> tuple[int | None, int | None]:
    """Token usage of a chat completion body, plain JSON or an SSE stream."""
    text = body.decode("utf-8", errors="replace")

    try:
        payload: Any = json.loads(text)
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        return usage_tokens(payload.get("usage"))

    found: tuple[int | None, int | None] = (None, None)

    for line in text.splitlines():
        if not line.startswith("data:"):
            continue

        data = line.removeprefix("data:").strip()

        if not data or data == "[DONE]":
            continue

        try:
            event = json.loads(data)
        except ValueError:
            continue

        if isinstance(event, dict) and event.get("usage"):
            found = usage_tokens(event["usage"])

    return found


def recorded_response(status_code: int, headers: dict[str, str], body: bytes) -> RecordedResponse:
    input_tokens, output_tokens = response_usage(body)
    kept = {name: headers[name] for name in REPLAYED_HEADERS if name in headers}
    return RecordedResponse(status_code, kept, body, input_tokens, output_tokens)
