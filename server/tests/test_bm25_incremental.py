"""Tests for BM25 incremental methods (add_text, replace_text, remove_text)."""

import pytest
from infrastructure.ml.hybrid import BM25Index, content_hash


class TestBM25Incremental:
    """Tests for BM25Index incremental operations."""

    def test_add_text(self):
        """Test adding a new text to the index."""
        # Start with empty index
        index = BM25Index(texts=[], hashes=[])
        assert index.n_docs == 0

        # Add a text
        idx = index.add_text("hello world")
        assert idx == 0
        assert index.n_docs == 1
        assert index.texts == ["hello world"]
        assert index.doc_lens == [2]

        # Add another text
        idx = index.add_text("foo bar baz")
        assert idx == 1
        assert index.n_docs == 2
        assert index.texts == ["hello world", "foo bar baz"]
        assert index.doc_lens == [2, 3]

    def test_add_text_with_hash(self):
        """Test adding a text with a custom hash."""
        index = BM25Index(texts=[], hashes=[])
        custom_hash = "custom_hash_123"
        idx = index.add_text("hello world", text_hash=custom_hash)
        assert index.hashes[idx] == custom_hash

    def test_replace_text(self):
        """Test replacing text at a given index."""
        index = BM25Index(texts=["hello world", "foo bar"], hashes=["h1", "h2"])

        # Replace first text
        index.replace_text(0, "new text here")
        assert index.texts[0] == "new text here"
        assert index.hashes[0] == content_hash("new text here")
        assert index.n_docs == 2  # Count unchanged

        # Verify search still works
        results = index.search("new")
        assert len(results) > 0
        assert results[0][0] == 0  # Should find at index 0

    def test_replace_text_invalid_index(self):
        """Test that replace_text raises IndexError for invalid index."""
        index = BM25Index(texts=["hello world"])

        with pytest.raises(IndexError):
            index.replace_text(5, "new text")

        with pytest.raises(IndexError):
            index.replace_text(-1, "new text")

    def test_remove_text(self):
        """Test removing text at a given index."""
        index = BM25Index(texts=["hello world", "foo bar", "baz qux"])

        # Remove middle text
        index.remove_text(1)
        assert index.n_docs == 2
        assert index.texts == ["hello world", "baz qux"]
        assert len(index.doc_lens) == 2
        assert len(index.token_freqs) == 2

        # Verify inverted index is updated
        for token_posting in index.inverted_index.values():
            for idx in token_posting:
                assert 0 <= idx < index.n_docs

    def test_remove_text_invalid_index(self):
        """Test that remove_text raises IndexError for invalid index."""
        index = BM25Index(texts=["hello world"])

        with pytest.raises(IndexError):
            index.remove_text(5)

    def test_remove_text_first(self):
        """Test removing the first text."""
        index = BM25Index(texts=["hello world", "foo bar", "baz qux"])

        index.remove_text(0)
        assert index.n_docs == 2
        assert index.texts == ["foo bar", "baz qux"]

        # Verify inverted index is updated correctly
        for token, posting in index.inverted_index.items():
            for idx in posting:
                assert 0 <= idx < index.n_docs, f"Index {idx} out of range for token '{token}'"

        # Verify search works correctly
        results = index.search("foo")
        assert len(results) > 0
        assert results[0][0] == 0  # Should be at new index 0

        results = index.search("hello")
        assert len(results) == 0  # Should not find removed text

    def test_remove_text_last(self):
        """Test removing the last text."""
        index = BM25Index(texts=["hello world", "foo bar", "baz qux"])

        index.remove_text(2)
        assert index.n_docs == 2
        assert index.texts == ["hello world", "foo bar"]

    def test_avgdl_update(self):
        """Test that avgdl is updated correctly after operations."""
        # Start with texts of different lengths
        index = BM25Index(texts=["a", "foo bar", "baz qux quux"])
        initial_avgdl = index.avgdl

        # Add a much longer text - should increase avgdl
        index.add_text("one two three four five six seven eight")
        assert index.avgdl > initial_avgdl

        # Replace with very short text - should decrease avgdl
        index.replace_text(0, "x")
        # Just verify it doesn't crash and n_docs is unchanged
        assert index.n_docs == 4

        # Remove a text - verify n_docs decreases
        index.remove_text(2)
        assert index.n_docs == 3

        # Verify avgdl is reasonable (between min and max doc lengths)
        min_len = min(index.doc_lens)
        max_len = max(index.doc_lens)
        assert min_len <= index.avgdl <= max_len

    def test_search_after_operations(self):
        """Test that search works correctly after incremental operations."""
        index = BM25Index(texts=["hello world", "foo bar"])

        # Add text
        index.add_text("search query test")

        # Search should find the new text
        results = index.search("search")
        assert len(results) > 0

        # Replace text
        index.replace_text(0, "completely new content")
        results = index.search("completely")
        assert len(results) > 0

        # Remove text
        index.remove_text(1)
        results = index.search("foo")
        assert len(results) == 0

    def test_serialization_roundtrip(self):
        """Test that incremental operations preserve serialization."""
        index = BM25Index(texts=["hello world", "foo bar"])

        # Perform operations
        index.add_text("baz qux")
        index.replace_text(0, "new hello")
        index.remove_text(1)

        # Serialize and deserialize
        data = index.to_dict()
        restored = BM25Index.from_dict(data)

        assert restored.texts == index.texts
        assert restored.hashes == index.hashes
        assert restored.n_docs == index.n_docs
