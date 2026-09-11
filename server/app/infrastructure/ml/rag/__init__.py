"""RAG helpers — backward-compatible re-exports from focused sub-modules.

This package replaces the monolithic ``rag.py`` (738 lines) with focused modules:

- ``rag_prompts``   — condense, decompose, summary, build_prompt
- ``rag_reranking`` — rerank_documents, deduplicate_docs, group_by_section
- ``rag_sources``   — extract_sources and metadata aggregation
- ``rag_formatting`` — format_docs, history_to_messages, CHARS_PER_TOKEN
- ``rag_relevance`` — check_relevance, filter_cited_sources
- ``rag_config``    — build_rag_settings
- ``rag_postprocess`` — is_not_found_answer, enrich_with_neighbors, etc.
- ``rag_steps``     — pipeline step functions for RagService.stream()
"""

# Re-export everything for backward compatibility
# All consumers doing ``from infrastructure.ml.rag import X`` continue to work.

from infrastructure.ml.rag.rag_formatting import (  # noqa: F401
    CHARS_PER_TOKEN,
    format_docs,
    history_to_messages,
)
from infrastructure.ml.rag.rag_prompts import (  # noqa: F401
    build_prompt,
    condense_question,
    decompose_question,
    update_rolling_summary,
)
from infrastructure.ml.rag.rag_relevance import (  # noqa: F401
    check_relevance,
    filter_cited_sources,
)
from infrastructure.ml.rag.rag_reranking import (  # noqa: F401
    deduplicate_docs,
    group_by_section,
    rerank_documents,
)
from infrastructure.ml.rag.rag_sources import extract_sources  # noqa: F401

# Re-export from domain for backward compatibility
from domain.services.rag_policy import (  # noqa: F401
    build_system_prompt,
    classify_question_breadth,
    is_out_of_domain,
    needs_decomposition,
)
