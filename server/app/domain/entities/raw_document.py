"""RawDocument — domain-owned parsed document, framework-independent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawDocument:
    """Parsed document content before chunking.

    Replaces langchain.schema.Document in domain/infra boundaries.
    Infrastructure adapters convert to/from LangChain types.
    """

    page_content: str = ""
    metadata: dict = field(default_factory=dict)
