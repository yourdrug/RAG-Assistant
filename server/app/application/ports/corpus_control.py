"""Prepare and publish a corpus snapshot across serving processes."""

from typing import Protocol


class CorpusControlPort(Protocol):
    async def is_ready(self) -> bool: ...

    async def status(self) -> dict: ...

    async def switch(self, endpoint: str, collection: str, expected_generation: str) -> dict: ...
