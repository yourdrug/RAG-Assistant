from collections.abc import Callable
from enum import StrEnum
from typing import overload

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

class Mode(StrEnum):
    JSON = "json_mode"
    TOOLS = "tool_call"
    MD_JSON = "markdown_json_mode"

class SyncCompletions:
    def create[T: BaseModel](self, *, response_model: type[T], **kwargs: object) -> T: ...

class AsyncCompletions:
    async def create[T: BaseModel](self, *, response_model: type[T], **kwargs: object) -> T: ...

class SyncChat:
    completions: SyncCompletions

class AsyncChat:
    completions: AsyncCompletions

class Instructor:
    client: OpenAI
    chat: SyncChat
    def on(self, name: str, handler: Callable[..., object]) -> None: ...

class AsyncInstructor:
    client: AsyncOpenAI
    chat: AsyncChat
    def on(self, name: str, handler: Callable[..., object]) -> None: ...

@overload
def from_openai(client: OpenAI, *, mode: Mode) -> Instructor: ...
@overload
def from_openai(client: AsyncOpenAI, *, mode: Mode) -> AsyncInstructor: ...
