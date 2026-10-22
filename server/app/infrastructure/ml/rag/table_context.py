"""Apply the domain table selection to a retrieval document."""

from copy import copy

from langchain.schema import Document

from domain.rag.evidence.table_context import focused_table_content


def focus_table_rows(doc: Document, query: str) -> Document:
    selection = focused_table_content(doc, query)

    if selection is None:
        return doc

    focused = copy(doc)
    focused.page_content, focused.metadata = selection
    return focused
