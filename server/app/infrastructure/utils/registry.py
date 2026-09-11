"""Ingestion registry utilities.

The JSON-file based registry has been replaced by a Postgres-backed
``IngestionRegistryRepository``.  This module now only provides the
``file_hash`` utility function used by the ingestion pipeline.

S3-only: hashes are based on ``FileItem.size_bytes`` and ``FileItem.last_modified``.
"""

import logging

from infrastructure.storage import FileItem

log = logging.getLogger("default")


def file_hash(source: FileItem) -> str:
    return f"{source.size_bytes}_{source.last_modified}"
