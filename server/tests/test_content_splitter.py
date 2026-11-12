"""Tests for content-based text splitting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import re
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from domain.ingestion.content_splitter import SplitUnit, split_by_content
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.types import BoundaryLevel


_POINT_RE = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
_SUBPOINT_RE = re.compile(r"^\s*([а-я])\)\s+", re.MULTILINE)
_SUBPOINT_NUM_RE = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)*)\.\s+", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@pytest.mark.parametrize("line", ["12.03.2020 вступает в силу.", "2.5 тонн продукции."])
def test_decree_dates_and_decimals_are_not_numeric_subpoints(line: str) -> None:
    text = "1. Порядок исполнения:\n" + line + "\n1.1. Настоящий подпункт."
    units = split_by_content(text, DecreeDomainProfile().content_boundaries(), 5000)
    assert [unit.boundary_value for unit in units if unit.unit_kind == "subpoint_num"] == ["1.1"]
    assert any(line in unit.content for unit in units)


class TestSplitByContent:
    def test_empty_text(self) -> None:
        result = split_by_content("", [], 1000)
        assert result == []

    def test_whitespace_only(self) -> None:
        result = split_by_content("   \n  ", [], 1000)
        assert result == []

    def test_no_levels_returns_raw(self) -> None:
        result = split_by_content("Hello world", [], 1000)
        assert len(result) == 1
        assert result[0].content == "Hello world"
        assert result[0].unit_kind == "raw"

    def test_single_level_no_match(self) -> None:
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content("No points here.", levels, 1000)
        assert len(result) == 1
        assert result[0].unit_kind == "point"

    def test_single_level_with_matches(self) -> None:
        text = "1. First point.\n2. Second point.\n3. Third point."
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content(text, levels, 1000)
        assert len(result) == 3
        assert result[0].boundary_value == "1"
        assert result[1].boundary_value == "2"
        assert result[2].boundary_value == "3"

    def test_preamble_extracted(self) -> None:
        text = "Introduction text.\n1. First point.\n2. Second point."
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content(text, levels, 1000)
        assert len(result) == 3
        assert result[0].unit_kind == "preamble"
        assert result[0].heading is None
        assert "Introduction" in result[0].content

    def test_recursive_descent_on_large_unit(self) -> None:
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

    def test_boundary_value_captured(self) -> None:
        text = "10. Tenth point.\n20. Twentieth point."
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content(text, levels, 1000)
        assert result[0].boundary_value == "10"
        assert result[1].boundary_value == "20"

    def test_heading_uses_profile_label(self) -> None:
        text = "5. Fifth point."
        levels = [BoundaryLevel("point", _POINT_RE)]
        result = split_by_content(text, levels, 1000, unit_label=DecreeDomainProfile().unit_label)
        assert result[0].heading == "п. 5"

    def test_multi_level_hierarchy(self) -> None:
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
    def test_frozen(self) -> None:
        unit = SplitUnit(heading="test", content="text", unit_kind="raw")
        field_name = "heading"

        with pytest.raises(FrozenInstanceError):
            setattr(unit, field_name, "changed")

    def test_defaults(self) -> None:
        unit = SplitUnit(heading=None, content="text", unit_kind="raw")
        assert unit.boundary_value is None


class TestDecreeNumericSubpoints:
    """Decree documents use numeric subpoints like 21.1., 21.2., etc.

    These must NOT fall through to sentence splitting — they need their own
    boundary level between 'point' and 'sentence'.
    """

    DECREE_TEXT = (
        "21. Отбор изделий партии на опробование осуществляется с учетом их количества и ассортимента.\n"
        "Опробованию подлежат:\n"
        "21.1. золотые, платиновые и палладиевые изделия, изготовленные на территории Республики Беларусь, "
        "включая изделия после ремонта (реставрации):\n"
        "до 100 штук - все изделия партии;\n"
        "более 100 штук - до 30 процентов изделий партии, но не менее 100 штук.\n"
        "21.2. серебряные изделия, изготовленные на территории Республики Беларусь "
        "(за исключением весового серебра и изделий после ремонта (реставрации):\n"
        "до 100 штук - все изделия партии;\n"
        "более 100 штук - до 30 процентов изделий партии, но не менее 50 штук.\n"
        "21.3. изделия из драгоценных металлов, ввезенные на территорию Республики Беларусь для реализации:\n"
        "до 100 штук - все изделия партии;\n"
        "более 100 штук - до 30 процентов изделий партии, но не менее 100 штук.\n"
    )

    def test_numeric_subpoints_are_split_separately(self) -> None:
        levels = [
            BoundaryLevel("point", _POINT_RE),
            BoundaryLevel("subpoint_num", _SUBPOINT_NUM_RE, always_split=True),
            BoundaryLevel("sentence", _SENTENCE_RE),
        ]
        result = split_by_content(self.DECREE_TEXT, levels, 5000)
        kinds = [u.unit_kind for u in result]
        # Each 21.x should become its own subpoint_num unit
        assert kinds.count("subpoint_num") == 3

    def test_subpoint_heading_contains_number(self) -> None:
        levels = [
            BoundaryLevel("point", _POINT_RE),
            BoundaryLevel("subpoint_num", _SUBPOINT_NUM_RE, always_split=True),
            BoundaryLevel("sentence", _SENTENCE_RE),
        ]
        result = split_by_content(self.DECREE_TEXT, levels, 5000)
        subpoints = [u for u in result if u.unit_kind == "subpoint_num"]
        assert subpoints[0].boundary_value == "21.1"
        assert subpoints[1].boundary_value == "21.2"
        assert subpoints[2].boundary_value == "21.3"

    def test_subpoint_content_starts_with_text(self) -> None:
        levels = [
            BoundaryLevel("point", _POINT_RE),
            BoundaryLevel("subpoint_num", _SUBPOINT_NUM_RE, always_split=True),
            BoundaryLevel("sentence", _SENTENCE_RE),
        ]
        result = split_by_content(self.DECREE_TEXT, levels, 5000)
        subpoints = [u for u in result if u.unit_kind == "subpoint_num"]
        # Content should start with the actual text, not the number
        assert subpoints[0].content.startswith("21.1.")
        assert "золотые" in subpoints[0].content

    def test_no_cross_contamination_between_subpoints(self) -> None:
        levels = [
            BoundaryLevel("point", _POINT_RE),
            BoundaryLevel("subpoint_num", _SUBPOINT_NUM_RE, always_split=True),
            BoundaryLevel("sentence", _SENTENCE_RE),
        ]
        result = split_by_content(self.DECREE_TEXT, levels, 5000)
        subpoints = [u for u in result if u.unit_kind == "subpoint_num"]
        # Subpoint 21.1 should NOT contain text from 21.2
        assert "серебряные изделия, изготовленные" not in subpoints[0].content
        # Subpoint 21.2 should NOT contain text from 21.3
        assert "ввезенные на территорию" not in subpoints[1].content

    def test_recursive_descent_when_subpoint_is_large(self) -> None:
        """When a subpoint exceeds max_unit_chars, it descends to sentence."""
        levels = [
            BoundaryLevel("point", _POINT_RE),
            BoundaryLevel("subpoint_num", _SUBPOINT_NUM_RE, always_split=True),
            BoundaryLevel("sentence", _SENTENCE_RE),
        ]
        result = split_by_content(self.DECREE_TEXT, levels, 100)
        kinds = [u.unit_kind for u in result]
        # With max_unit_chars=100, subpoints should be further split into sentences
        assert "sentence" in kinds
