"""Tests for infrastructure/registry.py -- file_hash utility.

The JSON-file based registry (load/save/is_already_indexed) has been
replaced by a Postgres-backed IngestionRegistryRepository.
This module now only tests the ``file_hash`` utility function for S3 FileItems.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.registry import file_hash  # noqa: E402
from infrastructure.storage import FileItem  # noqa: E402


# ---------------------------------------------------------------------------
# file_hash
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_fileitem_hash_contains_size_and_last_modified(self):
        item = FileItem(
            key="a.pdf",
            filename="a.pdf",
            size_bytes=1024,
            last_modified="12345",
            extension=".pdf",
        )
        h = file_hash(item)
        parts = h.split("_")
        assert len(parts) == 2
        assert parts[0] == "1024"
        assert parts[1] == "12345"

    def test_different_fileitems_different_hashes(self):
        item1 = FileItem(
            key="a.txt",
            filename="a.txt",
            size_bytes=100,
            last_modified="1000",
            extension=".txt",
        )
        item2 = FileItem(
            key="b.txt",
            filename="b.txt",
            size_bytes=200,
            last_modified="2000",
            extension=".txt",
        )
        assert file_hash(item1) != file_hash(item2)

    def test_same_fileitem_same_hash(self):
        item1 = FileItem(
            key="a.txt",
            filename="a.txt",
            size_bytes=100,
            last_modified="1000",
            extension=".txt",
        )
        item2 = FileItem(
            key="a.txt",
            filename="a.txt",
            size_bytes=100,
            last_modified="1000",
            extension=".txt",
        )
        assert file_hash(item1) == file_hash(item2)

    def test_fileitem_hash(self):
        item = FileItem(
            key="a.pdf",
            filename="a.pdf",
            size_bytes=1024,
            last_modified="12345",
            extension=".pdf",
        )
        h = file_hash(item)
        assert h == "1024_12345"

    def test_fileitem_hash_uses_size_and_last_modified(self):
        item = FileItem(
            key="x.docx",
            filename="x.docx",
            size_bytes=999,
            last_modified="99999",
            extension=".docx",
        )
        assert file_hash(item) == "999_99999"
