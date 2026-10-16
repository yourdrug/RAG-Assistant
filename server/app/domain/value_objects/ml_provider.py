"""Supported embedding and reranking providers."""

from enum import StrEnum


class MLProvider(StrEnum):
    TEI = "tei"
    DEEPINFRA = "deepinfra"
