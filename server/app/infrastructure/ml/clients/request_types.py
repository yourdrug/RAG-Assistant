"""Keyword contracts for infrastructure structured completion adapters."""

from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel


class CompletionRequest[T: BaseModel](TypedDict):
    response_model: type[T]
    model: str
    messages: list[dict[str, Any]]
    max_tokens: NotRequired[int]
    max_retries: NotRequired[int]
    temperature: NotRequired[float]
