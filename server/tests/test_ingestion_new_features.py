"""Tests for new ingestion features: splitting enrichment,
markdown front matter/hashtags/date patterns, RTF metadata.
"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from langchain.schema import Document

from infrastructure.ml.ingestion.splitting import (
    _enrich_final_chunks,
    _extract_first_sentence,
    classify_content_shape,
)


# ---------------------------------------------------------------------------
# Splitting enrichment
# ---------------------------------------------------------------------------


class TestSplittingEnrichment:
    def testclassify_content_shape_list(self) -> None:
        text = "- item one\n- item two\n- item three\n- item four"
        assert classify_content_shape(text) == "list"

    def testclassify_content_shape_definition(self) -> None:
        text = "Recursion — это функция которая вызывает саму себя."
        assert classify_content_shape(text) is None

    def testclassify_content_shape_procedure(self) -> None:
        text = "Шаг 1: Откройте файл.\nШаг 2: Нажмите сохранить."
        assert classify_content_shape(text) is None

    def testclassify_content_shape_none(self) -> None:
        text = "Just regular paragraph text with no special structure."
        assert classify_content_shape(text) is None

    def test_extract_first_sentence(self) -> None:
        text = "This is the first sentence. And this is the second."
        result = _extract_first_sentence(text)
        assert result is not None
        assert result.startswith("This is the first sentence.")

    def test_extract_first_sentence_empty(self) -> None:
        assert _extract_first_sentence("") is None

    def test_extract_first_sentence_no_boundary(self) -> None:
        text = "No period here"
        result = _extract_first_sentence(text)
        assert result == "No period here"

    def test_extract_first_sentence_truncates(self) -> None:
        text = "A" * 300
        result = _extract_first_sentence(text)
        assert result is not None
        assert len(result) <= 200

    def test_enrich_final_chunks_adds_metadata(self) -> None:
        chunks = [
            Document(
                page_content="Text with 2024-01-15 date and some numbers like 42.",
                metadata={"source": "test.txt"},
            ),
            Document(
                page_content="Second chunk without dates.",
                metadata={"source": "test.txt"},
            ),
        ]
        _enrich_final_chunks(chunks)

        # First chunk
        m1 = chunks[0].metadata
        assert m1["chunk_index"] == 1
        assert m1["total_chunks"] == 2
        assert m1["char_count"] > 0
        assert m1["word_count"] > 0
        assert m1["sentence_count"] >= 1
        assert m1["has_dates"] is True
        assert "2024-01-15" in m1["extracted_dates"]
        assert m1["has_numbers"] is True
        assert "first_sentence" in m1

        # Second chunk
        m2 = chunks[1].metadata
        assert m2["chunk_index"] == 2
        assert "has_dates" not in m2  # no dates in this chunk
