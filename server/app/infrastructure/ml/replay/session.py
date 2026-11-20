"""Benchmark-scoped activation of the LLM replay cache.

Replay is active only inside ``llm_replay_scope``: the session lives in a context
variable, which asyncio tasks and ``asyncio.to_thread`` inherit from the benchmark run.
Chat requests never enter the scope, so their LLM calls are never replayed.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
from typing import TYPE_CHECKING

from domain.benchmark.llm_replay import LLMReplayMode, LLMReplayStats


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from infrastructure.ml.replay.recording import RecordedResponse
    from infrastructure.ml.replay.store import ReplayStore


@dataclass
class ReplaySession:
    mode: LLMReplayMode
    store: ReplayStore
    stats: LLMReplayStats
    epoch: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)

    def hit(self, response: RecordedResponse) -> None:
        with self.lock:
            self.stats.hits += 1
            self.stats.saved_input_tokens += response.input_tokens or 0
            self.stats.saved_output_tokens += response.output_tokens or 0

    def miss(self) -> None:
        with self.lock:
            self.stats.misses += 1

    def recorded(self) -> None:
        with self.lock:
            self.stats.recorded += 1

    def store_error(self) -> None:
        with self.lock:
            self.stats.store_errors += 1


current_replay: ContextVar[ReplaySession | None] = ContextVar("llm_replay_session", default=None)


@asynccontextmanager
async def llm_replay_scope(
    store: ReplayStore, mode: LLMReplayMode, *, epoch: str = ""
) -> AsyncIterator[LLMReplayStats]:
    stats = LLMReplayStats(mode=mode)

    if mode == LLMReplayMode.OFF:
        yield stats
        return

    session = ReplaySession(mode=mode, store=store, stats=stats, epoch=epoch)
    token = current_replay.set(session)
    try:
        yield stats
    finally:
        current_replay.reset(token)
