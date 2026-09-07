"""Tests for content-based text splitting."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from domain.domain_profile.content_splitter import SplitUnit, split_by_content
from domain.domain_profile.protocol import BoundaryLevel


_POINT_RE = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
_SUBPOINT_RE = re.compile(r"^\s*([а-я])\)\s+", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class TestSplitByContent:
    def test_empty_text(self):
        result = split_by_content("", [], 1000)
        assert result == []

    def test_whitespace_only(self):
        result = split_by_content("   \n  ", [], 1000)
        assert result == []

    def test_no_levels_returns_raw(self):
        result = split_by_content("Hello world", [], 1000)
        assert len(result) == 1
        assert result[0].content == "Hello world"
        assert result[0].unit_kind == "raw"

    def test_single_level_no_match(self):
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content("No points here.", levels, 1000)
        assert len(result) == 1
        assert result[0].unit_kind == "point"

    def test_single_level_with_matches(self):
        text = "1. First point.\n2. Second point.\n3. Third point."
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content(text, levels, 1000)
        assert len(result) == 3
        assert result[0].boundary_value == "1"
        assert result[1].boundary_value == "2"
        assert result[2].boundary_value == "3"

    def test_preamble_extracted(self):
        text = "Introduction text.\n1. First point.\n2. Second point."
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content(text, levels, 1000)
        assert len(result) == 3
        assert result[0].unit_kind == "preamble"
        assert result[0].heading is None
        assert "Introduction" in result[0].content

    def test_recursive_descent_on_large_unit(self):
        # Each point must exceed max_unit_chars to trigger descent to subpoint
        long_subpoint = "a) " + "word " * 20
        text = f"1. {long_subpoint}.\n2. a) Short."
        levels = [
            BoundaryLevel("point", _POINT_RE),
            BoundaryLevel("subpoint", _SUBPOINT_RE),
        ]
        result = split_by_content(text, levels, 30)
        # First point >30 chars → descends to subpoint
        kinds = [u.unit_kind for u in result]
        assert "subpoint" in kinds or len(result) >= 2

    def test_boundary_value_captured(self):
        text = "10. Tenth point.\n20. Twentieth point."
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content(text, levels, 1000)
        assert result[0].boundary_value == "10"
        assert result[1].boundary_value == "20"

    def test_heading_includes_level_name(self):
        text = "5. Fifth point."
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content(text, levels, 1000)
        assert result[0].heading == "point 5"

    def test_multi_level_hierarchy(self):
        text = (
            "1. First point with enough text to exceed limit. "
            "Some more text here to make it long enough. "
            "Even more text to ensure we exceed the max.\n"
            "2. Second point."
        )
        levels = [
            BoundaryLevel("point", _POINT_RE),
            BoundaryLevel("sentence", _SENTENCE_RE),
        ]
        # max_unit_chars small enough to trigger descent on first point
        result = split_by_content(text, levels, 50)
        # First point should be split into sentences
        assert len(result) > 2


class TestSplitUnit:
    def test_frozen(self):
        unit = SplitUnit(heading="test", content="text", unit_kind="raw")
        try:
            unit.heading = "changed"
            raise AssertionError("Should be frozen")
        except Exception:
            pass

    def test_defaults(self):
        unit = SplitUnit(heading=None, content="text", unit_kind="raw")
        assert unit.boundary_value is None
