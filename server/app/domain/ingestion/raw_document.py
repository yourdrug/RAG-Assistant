"""RawDocument -- parsed document content before chunking, framework-independent.

Holds the full extracted text and metadata from a single file.  Serves as
the hand-off type between the parser (infrastructure) and the splitter
(domain service).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawDocument:
    """Parsed document content before chunking.

    Replaces langchain.schema.Document in domain/infra boundaries.
    Infrastructure adapters convert to/from LangChain types.
    """

    page_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
