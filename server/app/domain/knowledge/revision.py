"""Published revision of a document's chunks and reuse of unchanged embeddings.

A re-processed document is written as a hidden (staged) revision. Search and
citations keep using the published revision until every point of the staged one
is indexed; then one transaction makes the staged revision the published one.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.shared.hashing import content_hash


if TYPE_CHECKING:
    from collections.abc import Sequence


# Vector payload flag of a point that belongs to a revision not yet published.
STAGED_KEY = "staged"
# Outbox point hint: a published chunk whose dense vector may serve this chunk.
REUSE_POINT_KEY = "reuse_point_id"
# Outbox upsert payload key: the staged revision token whose points the entry uploads.
STAGED_REVISION_KEY = "staged_revision"
# Vector payload hash of the exact dense-model input, proving a vector is reusable.
EMBED_HASH_KEY = "embed_hash"


def embedding_input(content: str, embed_text: str | None) -> str:
    """The text the dense model embeds for a chunk."""
    return embed_text or content


def embedding_input_hash(content: str, embed_text: str | None) -> str:
    return content_hash(embedding_input(content, embed_text))


def match_reusable_chunks(
    published: Sequence[tuple[int, str]], staged_inputs: Sequence[str]
) -> list[int | None]:
    """Pair each staged chunk with a published chunk of identical dense input.

    Pairing is by occurrence: the k-th staged chunk with an input takes the k-th
    published chunk with that input, so repeated paragraphs never share one source.
    """
    pools: dict[str, deque[int]] = {}

    for chunk_id, text in published:
        pools.setdefault(text, deque()).append(chunk_id)

    return [pools[text].popleft() if pools.get(text) else None for text in staged_inputs]


@dataclass(frozen=True)
class SourceVersion:
    """Version of the external source a revision was made from (e.g. a storage file).

    It is recorded on the document only when the revision becomes the published one,
    so a document never claims a source version whose text search does not serve.
    """

    version: str
    revision: str | None = None


@dataclass(frozen=True)
class StagedRevision:
    """A hidden revision of a document's chunks, identified by an opaque token."""

    document_id: int
    token: str
    chunk_ids: tuple[int, ...]
    source_version: SourceVersion | None = None


@dataclass(frozen=True)
class RevisionSwitch:
    """Chunks that became published and the chunks they replaced."""

    published_ids: tuple[int, ...]
    replaced_ids: tuple[int, ...]
    # The source version the published revision was made from, if any.
    source_version: SourceVersion | None = None
    # The document domain the revision was classified into; the document takes it now.
    doc_domain: str | None = None
