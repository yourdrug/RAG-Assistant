"""Compatibility exports; use application.ingestion."""

from application.ingestion.parser import (
    DocumentParserPort,
    DocumentSplitterPort,
    FileMeta,
    SplitContext,
)


__all__ = ["DocumentParserPort", "DocumentSplitterPort", "FileMeta", "SplitContext"]
