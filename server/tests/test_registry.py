"""Tests for file_hash utility."""

from __future__ import annotations

from infrastructure.storage import FileItem


def file_hash(source: FileItem) -> str:
    return f"{source.size_bytes}_{source.last_modified}"


# ---------------------------------------------------------------------------
# file_hash
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_fileitem_hash_contains_size_and_last_modified(self) -> None:
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

    def test_different_fileitems_different_hashes(self) -> None:
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

    def test_same_fileitem_same_hash(self) -> None:
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

    def test_fileitem_hash(self) -> None:
        item = FileItem(
            key="a.pdf",
            filename="a.pdf",
            size_bytes=1024,
            last_modified="12345",
            extension=".pdf",
        )
        h = file_hash(item)
        assert h == "1024_12345"

    def test_fileitem_hash_uses_size_and_last_modified(self) -> None:
        item = FileItem(
            key="x.docx",
            filename="x.docx",
            size_bytes=999,
            last_modified="99999",
            extension=".docx",
        )
        assert file_hash(item) == "999_99999"
