"""Chunk entity -- a single searchable document fragment stored in the vector store.

Carries the text content and metadata (source file, document_id, visibility,
owner/group ACL).  Created by the ingestion pipeline and consumed by the
retriever during RAG queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str = ""
    metadata: dict = field(default_factory=dict)
    score: float | None = None
