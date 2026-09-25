"""RAG helpers — focused sub-modules.

- ``rag_prompts``    — condense, decompose, summary, build_prompt
- ``rag_reranking``  — rerank_documents, deduplicate_docs, group_by_section
- ``rag_sources``    — extract_sources and metadata aggregation
- ``rag_formatting`` — format_docs, history_to_messages, CHARS_PER_TOKEN
- ``rag_relevance``  — check_relevance, filter_cited_sources
- ``rag_config``     — build_rag_settings
- ``rag_postprocess`` — is_not_found_answer, enrich_with_neighbors, etc.
- ``rag_steps``      — pipeline step functions for RagService.stream()
"""
