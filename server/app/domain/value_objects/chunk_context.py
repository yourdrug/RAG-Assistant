"""Context needed to interpret a chunk independently of its retrieval backend."""

from collections.abc import Mapping
from copy import deepcopy


TABLE_CONTEXT_MAX_CHUNKS = 32


CHUNK_CONTEXT_FIELDS = frozenset(
    {
        "section",
        "heading",
        "heading_level",
        "parent_section",
        "doc_title",
        "doc_type",
        "doc_date",
        "content_type",
        "page",
        "page_start",
        "page_end",
        "pages",
        "unit_kind",
        "unit_heading_text",
        "parent_units",
        "article_number",
        "chapter_number",
        "section_number",
        "clause_number",
        "point_number",
        "subpoint_number",
        "subpoint_num_number",
        "step_number",
        "step_numbers",
        "table_id",
        "table_header",
        "table_row_start",
        "table_row_end",
        "table_row_key",
        "domain_metadata",
    }
)


def extract_chunk_context(metadata: Mapping) -> dict:
    """Copy interpretation fields, excluding identity, ACL and temporal state."""
    return {key: deepcopy(value) for key, value in metadata.items() if key in CHUNK_CONTEXT_FIELDS}
