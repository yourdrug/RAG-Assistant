"""Port for situating notes that enrich only the dense embedding input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ContextualizationRequest:
    """One chunk with its bounded document surroundings.

    ``document_key`` and ``content_key`` identify the note for caching, so
    re-indexing an unchanged document does not pay for the same calls again.
    All text fields are untrusted document content.
    """

    document_key: str
    content_key: str
    doc_title: str
    section: str
    previous: str
    content: str
    following: str


class ChunkContextualizer(Protocol):
    async def summarize(self, requests: Sequence[ContextualizationRequest]) -> list[str | None]:
        """Return one note per request; ``None`` marks an explicit, logged degradation."""
        ...
