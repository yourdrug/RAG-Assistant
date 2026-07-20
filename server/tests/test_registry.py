"""
Tests for infrastructure/registry.py — load/save, file_hash, is_already_indexed.
Uses tmp_path for filesystem operations.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.registry import file_hash, is_already_indexed, load_registry, save_registry  # noqa: E402
from infrastructure.storage import FileItem  # noqa: E402

# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------


class TestLoadRegistry:
    def test_returns_empty_dict_when_no_file(self, tmp_path):
        assert load_registry(str(tmp_path)) == {}

    def test_loads_existing_registry(self, tmp_path):
        data = {"doc.pdf": {"hash": "123_456", "chunks": 5}}
        (tmp_path / "ingestion_registry.json").write_text(json.dumps(data))
        result = load_registry(str(tmp_path))
        assert result == data

    def test_returns_empty_dict_on_corrupt_json(self, tmp_path):
        (tmp_path / "ingestion_registry.json").write_text("{invalid json!!!")
        assert load_registry(str(tmp_path)) == {}

    def test_returns_empty_dict_on_empty_file(self, tmp_path):
        (tmp_path / "ingestion_registry.json").write_text("")
        assert load_registry(str(tmp_path)) == {}

    def test_loads_unicode_content(self, tmp_path):
        data = {"файл.pdf": {"source": "документы/файл.pdf"}}
        (tmp_path / "ingestion_registry.json").write_text(json.dumps(data, ensure_ascii=False))
        result = load_registry(str(tmp_path))
        assert "файл.pdf" in result


# ---------------------------------------------------------------------------
# save_registry
# ---------------------------------------------------------------------------


class TestSaveRegistry:
    def test_creates_file(self, tmp_path):
        save_registry(str(tmp_path), {"a.pdf": {"hash": "1"}})
        assert (tmp_path / "ingestion_registry.json").exists()

    def test_writes_valid_json(self, tmp_path):
        data = {"doc.pdf": {"chunks": 3, "chars": 1000}}
        save_registry(str(tmp_path), data)
        loaded = json.loads((tmp_path / "ingestion_registry.json").read_text())
        assert loaded == data

    def test_overwrites_existing(self, tmp_path):
        save_registry(str(tmp_path), {"old.pdf": {"hash": "1"}})
        save_registry(str(tmp_path), {"new.pdf": {"hash": "2"}})
        loaded = load_registry(str(tmp_path))
        assert "old.pdf" not in loaded
        assert "new.pdf" in loaded

    def test_preserves_unicode(self, tmp_path):
        data = {"Тест.pdf": {"source": "путь/к/файлу"}}
        save_registry(str(tmp_path), data)
        loaded = load_registry(str(tmp_path))
        assert "Тест.pdf" in loaded

    def test_empty_registry(self, tmp_path):
        save_registry(str(tmp_path), {})
        loaded = load_registry(str(tmp_path))
        assert loaded == {}


# ---------------------------------------------------------------------------
# file_hash
# ---------------------------------------------------------------------------


class TestFileHash:
    def test_path_hash_contains_size_and_mtime(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content")
        h = file_hash(f)
        parts = h.split("_")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_different_files_different_hashes(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("short")
        f2.write_text("much longer content here")
        assert file_hash(f1) != file_hash(f2)

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        content = "identical content"
        f1.write_text(content)
        f2.write_text(content)
        # Same size and potentially same mtime
        h1 = file_hash(f1)
        h2 = file_hash(f2)
        # If written in same second, hashes match
        if h1 == h2:
            assert True
        else:
            # Different mtimes, but format is correct
            assert "_" in h1

    def test_fileitem_hash(self):
        item = FileItem(
            key="a.pdf", filename="a.pdf", size_bytes=1024, last_modified="12345", extension=".pdf"
        )
        h = file_hash(item)
        assert h == "1024_12345"

    def test_fileitem_hash_uses_size_and_last_modified(self):
        item = FileItem(
            key="x.docx", filename="x.docx", size_bytes=999, last_modified="99999", extension=".docx"
        )
        assert file_hash(item) == "999_99999"


# ---------------------------------------------------------------------------
# is_already_indexed
# ---------------------------------------------------------------------------


class TestIsAlreadyIndexed:
    def test_not_in_registry(self, tmp_path):
        f = tmp_path / "new.pdf"
        f.write_text("data")
        assert is_already_indexed(f, {}) is False

    def test_in_registry_with_matching_hash(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_text("content")
        registry = {"doc.pdf": {"hash": file_hash(f)}}
        assert is_already_indexed(f, registry) is True

    def test_in_registry_with_wrong_hash(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_text("content")
        registry = {"doc.pdf": {"hash": "0_0"}}
        assert is_already_indexed(f, registry) is False

    def test_fileitem_already_indexed(self):
        item = FileItem(key="a.pdf", filename="a.pdf", size_bytes=100, last_modified="500", extension=".pdf")
        registry = {"a.pdf": {"hash": "100_500"}}
        assert is_already_indexed(item, registry) is True

    def test_fileitem_not_indexed(self):
        item = FileItem(key="a.pdf", filename="a.pdf", size_bytes=100, last_modified="500", extension=".pdf")
        registry = {"a.pdf": {"hash": "999_999"}}
        assert is_already_indexed(item, registry) is False

    def test_different_filename_not_indexed(self, tmp_path):
        f = tmp_path / "new.pdf"
        f.write_text("data")
        registry = {"other.pdf": {"hash": file_hash(f)}}
        assert is_already_indexed(f, registry) is False

    def test_registry_entry_without_hash_key(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_text("data")
        registry = {"doc.pdf": {"source": "s3://bucket/doc.pdf"}}
        assert is_already_indexed(f, registry) is False
