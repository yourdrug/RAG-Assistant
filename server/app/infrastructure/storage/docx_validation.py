"""Compatible exports for shared DOCX archive validation."""

from application.documents.docx_validation import (
    DOCX_MAX_ARCHIVE_ENTRIES,
    DOCX_MAX_COMPRESSION_RATIO,
    DOCX_MAX_UNCOMPRESSED_BYTES,
    DOCX_READ_CHUNK_BYTES,
    ZIP_ENCRYPTED_FLAG,
    validate_docx_archive,
    verify_docx_entry_size,
)


__all__ = [
    "DOCX_MAX_ARCHIVE_ENTRIES",
    "DOCX_MAX_COMPRESSION_RATIO",
    "DOCX_MAX_UNCOMPRESSED_BYTES",
    "DOCX_READ_CHUNK_BYTES",
    "ZIP_ENCRYPTED_FLAG",
    "validate_docx_archive",
    "verify_docx_entry_size",
]
