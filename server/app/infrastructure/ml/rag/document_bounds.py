"""Context note when a cited article or table is absent from the act the question names.

Without it a missed search reads as «information not found», and a false premise
(«статья 40» of a law with 33 articles) is never corrected. The note is built only
from the act's own outline, read under the reader's access, and attached to a
chunk of that act the reader was already authorized to see.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from langchain.schema import Document as LCDocument

from domain.knowledge.chunk import CHUNK_CONTEXT_FIELDS
from domain.rag.act_references import referenced_act
from domain.rag.document_outline import cited_units, missing_unit_note
from domain.rag.ports import DocumentOutlineRepository
from domain.rag.ranking import RETAINED_AS_REQUESTED
from infrastructure.ml.evidence_profiles.profiles import current_evidence_profile


if TYPE_CHECKING:
    from domain.identity.user_context import UserContext


log = logging.getLogger("default")

OUTLINE_UNIT_KIND = "document_outline"
# Chunk-level fields that would make the note look like a provision of the template chunk.
CHUNK_IDENTITY_FIELDS = frozenset({"chunk_id", "chunk_index", "content_hash", "node_id", "table_id"})
MAX_OUTLINE_DOCUMENTS = 2


def named_act_chunks(docs: list[LCDocument], acts: tuple[str, ...]) -> list[LCDocument]:
    """One authorized chunk per document titled with an act the question names."""
    by_document: dict[object, LCDocument] = {}

    for doc in docs:
        document_id = doc.metadata.get("document_id")

        if document_id is not None and document_id not in by_document and referenced_act(doc.metadata, acts):
            by_document[document_id] = doc

    return list(by_document.values())[:MAX_OUTLINE_DOCUMENTS]


def outline_note_document(
    template: LCDocument, note: str, *, section: str = "Состав документа"
) -> LCDocument:
    metadata = {
        key: value
        for key, value in template.metadata.items()
        if key not in CHUNK_CONTEXT_FIELDS and key not in CHUNK_IDENTITY_FIELDS
    }
    title = template.metadata.get("doc_title")
    metadata.update(
        {
            "doc_title": title,
            "doc_date": template.metadata.get("doc_date"),
            "section": section,
            "unit_kind": OUTLINE_UNIT_KIND,
            "content_hash": hashlib.sha256(note.encode()).hexdigest()[:16],
            RETAINED_AS_REQUESTED: True,
        }
    )
    return LCDocument(page_content=note, metadata=metadata)


async def add_missing_unit_notes(
    docs: list[tuple[LCDocument, float]],
    candidates: list[LCDocument],
    question: str,
    chunk_search: object,
    user: UserContext,
) -> list[tuple[LCDocument, float]]:
    """Put first a note for each cited article or table the named act does not contain."""
    units = cited_units(question)

    if not units or not isinstance(chunk_search, DocumentOutlineRepository):
        return docs

    acts = current_evidence_profile().named_acts(question)

    if not acts:
        return docs

    present = {doc.page_content for doc, _ in docs}
    notes: list[tuple[LCDocument, float]] = []
    top_score = max((score for _, score in docs), default=1.0)

    for template in named_act_chunks([*(doc for doc, _ in docs), *candidates], acts):
        outline = await chunk_search.get_document_outline(int(template.metadata["document_id"]), user=user)
        title = str(template.metadata.get("doc_title") or template.metadata.get("source") or "")

        for unit in units:
            note = missing_unit_note(unit, outline, title)

            if note and note not in present:
                log.info(
                    "Cited %s %s absent from document %s",
                    unit.kind,
                    unit.value,
                    template.metadata["document_id"],
                )
                notes.append((outline_note_document(template, note), top_score))
                present.add(note)

    return [*notes, *docs]
