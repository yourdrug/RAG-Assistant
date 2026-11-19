"""Reuse of recorded LLM responses across benchmark runs.

A benchmark run may replay a recorded LLM response when the request is byte-for-byte
the same (model, parameters and prompt). Only stages whose input changed are paid for.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class LLMReplayMode(StrEnum):
    # Every LLM request goes to the provider; nothing is read or written.
    OFF = "off"
    # A recorded response is reused; a miss is sent to the provider and recorded.
    RECORD = "record"
    # Only recorded responses are used; a miss fails that LLM call instead of paying.
    REPLAY_ONLY = "replay_only"
    # Every request goes to the provider and overwrites the recorded response.
    REFRESH = "refresh"


@dataclass
class LLMReplayStats:
    """Counters of one benchmark run; tokens are those of the reused responses."""

    mode: LLMReplayMode = LLMReplayMode.OFF
    hits: int = 0
    misses: int = 0
    recorded: int = 0
    store_errors: int = 0
    saved_input_tokens: int = 0
    saved_output_tokens: int = 0

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "mode": self.mode.value}
