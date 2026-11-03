"""Streaming response body used by S3 cancellation and lifecycle tests."""

from __future__ import annotations

import asyncio
from typing import Self


class Body:
    def __init__(self, blocks: list[bytes], error: BaseException | None = None) -> None:
        self.blocks = iter(blocks)
        self.error = error
        self.closed = False
        self.read_sizes: list[int] = []
        self.started = asyncio.Event()
        self.release: asyncio.Event | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args
        self.closed = True

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        self.started.set()

        if self.release is not None:
            await self.release.wait()

        block = next(self.blocks, None)

        if block is not None:
            return block

        if self.error:
            raise self.error

        return b""
