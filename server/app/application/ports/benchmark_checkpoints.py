"""Durable stage storage for resumable benchmark execution."""

from typing import Protocol


class BenchmarkCheckpoints(Protocol):
    async def load(self, key: str) -> object | None: ...

    async def save(self, key: str, value: object) -> None: ...
