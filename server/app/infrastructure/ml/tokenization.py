"""Local embedding token counters; never download models during application startup."""

from __future__ import annotations

from functools import lru_cache
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import Tokenizer


if TYPE_CHECKING:
    from domain.ingestion.ports import TokenCounter


CHARS_PER_TOKEN = 4

log = logging.getLogger("default")


class HfTokenCounter:
    def __init__(self, path: str | Path) -> None:
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True).ids)


class CharEstimateTokenCounter:
    def count(self, text: str) -> int:
        return math.ceil(len(text) / CHARS_PER_TOKEN)


@lru_cache(maxsize=8)
def build_token_counter(path: str) -> TokenCounter:
    resolved = Path(path)

    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[4] / resolved

    if resolved.is_file():
        return HfTokenCounter(resolved)

    log.warning("Embedding tokenizer file is missing; chunking uses a character estimate")
    return CharEstimateTokenCounter()
