"""Compatibility exports; use application.chunks."""

from application.chunks.access import (
    load_doc_for_add,
    load_doc_for_edit,
)


__all__ = ["load_doc_for_add", "load_doc_for_edit"]
