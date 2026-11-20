"""Port for situating notes that enrich only the dense embedding input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ContextualizationRequest:
    """One chunk with its bounded document surroundings.

    ``input_key`` hashes exactly the prompt inputs (title, section, neighbors,
    content), so re-indexing pays only for chunks whose surroundings changed.
    All text fields are untrusted document content.
    """

    input_key: str
    doc_title: str
    section: str
    previous: str
    content: str
    following: str


class ChunkContextualizer(Protocol):
    async def summarize(self, requests: Sequence[ContextualizationRequest]) -> list[str | None]:
        """Return one note per request; ``None`` marks an explicit, logged degradation."""
        ...
