"""Bound DOCX ZIP resources before python-docx decompresses any member."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, BinaryIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile
import zlib


if TYPE_CHECKING:
    from pathlib import Path
    from zipfile import ZipInfo


DOCX_MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
DOCX_MAX_ARCHIVE_ENTRIES = 1000
DOCX_MAX_COMPRESSION_RATIO = 200
DOCX_READ_CHUNK_BYTES = 64 * 1024
ZIP_ENCRYPTED_FLAG = 0x1


def verify_docx_entry_size(archive: ZipFile, entry: ZipInfo) -> None:
    """Check real expansion in bounded reads, including forged ZIP size fields."""
    if entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
        msg = "DOCX ZIP compression must be stored or deflated"
        raise ValueError(msg)

    if entry.flag_bits & ZIP_ENCRYPTED_FLAG:
        msg = "Encrypted DOCX ZIP entries are not supported"
        raise ValueError(msg)

    bounded_entry = copy(entry)
    # ZipExtFile truncates output to file_size. Raise that artificial boundary
    # while streaming so an underreported size cannot conceal extra expansion.
    bounded_entry.file_size = DOCX_MAX_UNCOMPRESSED_BYTES + 1
    actual_size = 0
    with archive.open(bounded_entry) as member:
        while data := member.read(DOCX_READ_CHUNK_BYTES):
            actual_size += len(data)

            if actual_size > entry.file_size:
                msg = "DOCX ZIP entry has inconsistent uncompressed size"
                raise ValueError(msg)

    if actual_size != entry.file_size:
        msg = "DOCX ZIP entry has inconsistent uncompressed size"
        raise ValueError(msg)


def validate_docx_archive(source: Path | BinaryIO) -> None:
    """Reject unsafe metadata first, then verify actual sizes without buffering files."""
    try:
        with ZipFile(source) as archive:
            entries = archive.infolist()

            if len(entries) > DOCX_MAX_ARCHIVE_ENTRIES:
                msg = f"DOCX archive exceeds {DOCX_MAX_ARCHIVE_ENTRIES} entries"
                raise ValueError(msg)

            total_size = 0

            for entry in entries:
                total_size += entry.file_size

                if total_size > DOCX_MAX_UNCOMPRESSED_BYTES:
                    msg = "DOCX archive exceeds the maximum uncompressed size (100 MiB)"
                    raise ValueError(msg)

                if entry.file_size > DOCX_MAX_COMPRESSION_RATIO * entry.compress_size:
                    msg = "DOCX archive exceeds the maximum compression ratio (200:1)"
                    raise ValueError(msg)

            for entry in entries:
                verify_docx_entry_size(archive, entry)
    except (BadZipFile, zlib.error, EOFError) as exc:
        msg = "File content is not a valid ZIP-archive (.docx)"
        raise ValueError(msg) from exc
