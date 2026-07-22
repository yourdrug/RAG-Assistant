"""Chunk entity — domain representation of a retrieved document chunk."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str = ""
    metadata: dict = field(default_factory=dict)
    score: float | None = None
