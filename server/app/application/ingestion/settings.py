"""Ingestion settings port — abstracts configuration for the ingestion pipeline."""

from __future__ import annotations

from typing import Protocol


class IngestionSettingsPort(Protocol):
    @property
    def s3_bucket(self) -> str: ...

    @property
    def embed_dim(self) -> int: ...

    @property
    def collection_name(self) -> str: ...

    @property
    def hybrid_enabled(self) -> bool: ...

    @property
    def document_domain_marker_threshold(self) -> float: ...

    @property
    def tei_embed_url(self) -> str: ...

    @property
    def qdrant_url(self) -> str: ...
