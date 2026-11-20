"""Read a short document on the question's subject whole when reranking rejected every candidate.

The relevance gate and the generator then see the entire source with a note that
it is complete, so «the source does not set this value; it describes …» rests on
the document's bounds instead of on a missed search.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.schema import Document as LCDocument

from domain.rag.topic_document import complete_indexes, names_subject, subject_stems
from infrastructure.ml.rag.chunk_metadata import chunk_result_metadata
from infrastructure.ml.rag.document_bounds import outline_note_document


if TYPE_CHECKING:
    from domain.identity.user_context import UserContext
    from domain.knowledge.chunk import ChunkRecord
    from domain.rag.ports import ChunkRetrievalRepository


log = logging.getLogger("default")

# A document read whole fits the context next to the note; longer ones are never claimed complete.
MAX_TOPIC_DOCUMENT_CHUNKS = 12
MAX_TOPIC_DOCUMENTS_CHECKED = 3
TOPIC_DOCUMENT_SCORE = 0.0
COMPLETE_DOCUMENT_NOTE = "complete_document_note"


def candidate_documents(candidates: list[LCDocument]) -> list[int]:
    ids = (doc.metadata.get("document_id") for doc in candidates)
    return list(dict.fromkeys(int(i) for i in ids if i is not None))[:MAX_TOPIC_DOCUMENTS_CHECKED]


def opening(chunks: list[ChunkRecord]) -> str:
    first = chunks[0]
    return f"{first.doc_title or ''}\n{first.filename}\n{first.content.split(chr(10), 1)[0]}"


async def topic_document_fallback(
    question: str,
    candidates: list[LCDocument],
    chunk_search: ChunkRetrievalRepository | None,
    user: UserContext,
) -> list[tuple[LCDocument, float]]:
    """The one short candidate document whose opening names the question's subject, read whole."""
    stems = subject_stems(question)

    if chunk_search is None or not stems:
        return []

    matches: list[list[ChunkRecord]] = []

    for document_id in candidate_documents(candidates):
        chunks = await chunk_search.get_neighbors(document_id, 0, MAX_TOPIC_DOCUMENT_CHUNKS, user=user)

        if complete_indexes([c.chunk_index for c in chunks], MAX_TOPIC_DOCUMENT_CHUNKS) and names_subject(
            opening(chunks), stems
        ):
            matches.append(chunks)

    if len(matches) != 1:
        return []

    chunks = matches[0]
    docs = [LCDocument(page_content=c.content, metadata=chunk_result_metadata(c)) for c in chunks]
    title = chunks[0].doc_title or chunks[0].filename
    note = (
        f"Документ «{title}» приведён полностью: {len(chunks)} фрагментов, других положений в нём нет. "
        "Чего нет в этих фрагментах, документ не устанавливает."
    )
    log.info("Topic document %s read whole after reranking rejected all candidates", chunks[0].document_id)
    note_doc = outline_note_document(docs[0], note, section="Полный текст документа")
    note_doc.metadata[COMPLETE_DOCUMENT_NOTE] = True
    return [(doc, TOPIC_DOCUMENT_SCORE) for doc in [note_doc, *docs]]


def complete_topic_context(docs: list[tuple[LCDocument, float]] | list[LCDocument]) -> bool:
    """Whether the context is a topic document the fallback read whole."""
    return any(
        (item[0] if isinstance(item, tuple) else item).metadata.get(COMPLETE_DOCUMENT_NOTE) for item in docs
    )
