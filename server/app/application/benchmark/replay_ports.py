"""Port for reusing recorded LLM responses within one benchmark run."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol

from domain.benchmark.llm_replay import LLMReplayStats


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager

    from domain.benchmark.llm_replay import LLMReplayMode


class LLMReplayScope(Protocol):
    """Activate replay for LLM calls made inside the block; yields the run's counters."""

    def __call__(self, mode: LLMReplayMode) -> AbstractAsyncContextManager[LLMReplayStats]: ...


@asynccontextmanager
async def replay_disabled(mode: LLMReplayMode) -> AsyncIterator[LLMReplayStats]:
    """Scope used when no replay store is wired: every LLM call goes to the provider."""
    del mode
    yield LLMReplayStats()
