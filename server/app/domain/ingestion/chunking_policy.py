"""Validated chunking policy declared by a document profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.types import ConfigDefault


@dataclass(frozen=True)
class ChunkingPolicy:
    target_tokens: int
    max_tokens: int
    min_tokens: int = 25
    overlap_tokens: int = 0
    pack_siblings: bool = True
    context_depth: int = 2
    include_doc_title: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.target_tokens <= self.max_tokens:
            msg = "Target chunk tokens must be positive and must not exceed maximum chunk tokens"
            raise ValueError(msg)

        if not 0 <= self.overlap_tokens < self.target_tokens:
            msg = "Overlap must be non-negative and smaller than target tokens"
            raise ValueError(msg)

        if self.min_tokens < 0 or self.context_depth < 0:
            msg = "Minimum tokens and context depth must be non-negative"
            raise ValueError(msg)


def profile_policy(
    key: str,
    defaults: list[ConfigDefault],
    settings: DomainSettingsPort | None,
    *,
    overlap_tokens: int | None = None,
) -> ChunkingPolicy:
    """Build a profile policy; ``overlap_tokens=None`` means the profile forbids overlap."""
    values = {item.key: int(item.value) for item in defaults if item.key.endswith("_tokens")}

    def read(suffix: str, fallback: int) -> int:
        name = f"{key}_{suffix}"
        default = values.get(name, fallback)

        if settings is None:
            return default

        try:
            value = int(settings.get(name, domain_key=key))
        except (KeyError, TypeError, ValueError):
            return default
        return value if value > 0 or (suffix == "chunk_overlap_tokens" and value == 0) else default

    return ChunkingPolicy(
        target_tokens=read("chunk_target_tokens", 350),
        max_tokens=read("chunk_max_tokens", 512),
        overlap_tokens=0 if overlap_tokens is None else read("chunk_overlap_tokens", overlap_tokens),
    )
