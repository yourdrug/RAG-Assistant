"""Optional LLM situating notes for dense embeddings (chunking plan 4.3).

Runs after splitting and before any write transaction: notes are generated
first, then the usual short UoW persists chunks and the vector outbox. A note
changes only ``embed_text``; source content, prompts, citations and ACL
metadata never see it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.indexing.ports.chunk_contextualizer import ContextualizationRequest
from domain.ingestion.embed_text import MAX_EMBED_TOKENS, estimate_tokens, with_context_summary
from domain.shared.hashing import content_hash


if TYPE_CHECKING:
    from application.indexing.ports.chunk_contextualizer import ChunkContextualizer
    from domain.ingestion.ports import TokenCounter
    from domain.ingestion.raw_document import RawDocument

log = logging.getLogger("default")

NEIGHBOR_CONTEXT_CHARS = 600


def contextualization_requests(chunks: list[RawDocument]) -> list[ContextualizationRequest]:
    """Describe every chunk with its document identity and immediate neighbors."""
    by_source: dict[str, list[int]] = {}

    for index, chunk in enumerate(chunks):
        by_source.setdefault(str(chunk.metadata.get("source", "")), []).append(index)

    requests: list[ContextualizationRequest | None] = [None] * len(chunks)

    for indexes in by_source.values():
        bodies = [chunks[index].page_content for index in indexes]
        document_key = content_hash("\n\x1e".join(bodies))

        for position, index in enumerate(indexes):
            metadata = chunks[index].metadata
            requests[index] = ContextualizationRequest(
                document_key=document_key,
                content_key=content_hash(bodies[position]),
                doc_title=str(metadata.get("doc_title") or ""),
                section=str(metadata.get("section") or ""),
                previous=bodies[position - 1][-NEIGHBOR_CONTEXT_CHARS:] if position > 0 else "",
                content=bodies[position],
                following=bodies[position + 1][:NEIGHBOR_CONTEXT_CHARS] if position + 1 < len(bodies) else "",
            )

    return [request for request in requests if request is not None]


class ChunkContextualization:
    """Add generated notes to dense inputs within the embedding token limit."""

    def __init__(
        self,
        contextualizer: ChunkContextualizer,
        token_counter: TokenCounter | None = None,
        *,
        max_tokens: int = MAX_EMBED_TOKENS,
    ) -> None:
        self._contextualizer = contextualizer
        self._measure = token_counter.count if token_counter is not None else estimate_tokens
        self._max_tokens = max_tokens

    async def apply(self, chunks: list[RawDocument]) -> None:
        if not chunks:
            return

        summaries = await self._contextualizer.summarize(contextualization_requests(chunks))

        if len(summaries) != len(chunks):
            message = "Contextualizer must return one result per chunk"
            raise RuntimeError(message)

        annotated = over_budget = 0

        for chunk, summary in zip(chunks, summaries, strict=True):
            if not summary:
                continue

            combined = with_context_summary(
                chunk.embed_text or chunk.page_content,
                summary,
                max_tokens=self._max_tokens,
                measure=self._measure,
            )

            if combined is None:
                over_budget += 1
                continue

            chunk.embed_text = combined
            annotated += 1

        missing = len(chunks) - annotated - over_budget

        if missing or over_budget:
            log.warning(
                "Chunk contextualization degraded: annotated=%d without_note=%d over_budget=%d",
                annotated,
                missing,
                over_budget,
            )
        else:
            log.info("Chunk contextualization: annotated=%d", annotated)
