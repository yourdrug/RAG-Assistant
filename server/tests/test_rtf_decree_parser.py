"""Tests for the RTF decree parser and fingerprint-based routing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from infrastructure.ml.rtf_decree_parser import parse_decree_rtf
from domain.domain_profile.profiles.decree import DecreeDomainProfile


class FakeSettings:
    def get(self, key: str, domain_key: str = "") -> str:
        defaults = {"max_unit_chars": "800", "fingerprint_min_points": "1"}
        return defaults.get(key, "0")


_DECREE_RTF = (
    r"{\rtf1\ansi УКАЗ ПРЕЗИДЕНТА РЕСПУБЛИКИ БЕЛАРУСЬ\par "
    r"№ 123 от 1 января 2026 г.\par "
    r"О мерах по регулированию\par\par "
    r"ПОСТАНОВЛЯЮ:\par\par "
    r"1. Принять предложенные меры по совершенствованию.\par\par "
    r"2. Контроль за исполнением возложить на Министерство.\par}"
)

_PLAIN_RTF = r"{\rtf1\ansi Обычное письмо без признаков указа.\par Вторая строка текста.\par}"


def _decree_profile() -> DecreeDomainProfile:
    return DecreeDomainProfile(settings=FakeSettings())


def test_parse_decree_rtf_extracts_header_metadata_and_units(tmp_path):
    file_path = tmp_path / "ukaz.rtf"
    file_path.write_text(_DECREE_RTF, encoding="utf-8")

    units, doc_metadata = parse_decree_rtf(file_path, _decree_profile(), FakeSettings())

    assert doc_metadata.get("decree_number") == "123"
    kinds = [u.unit_kind for u in units]
    assert "preamble" in kinds
    assert "point" in kinds
    point_values = [u.boundary_value for u in units if u.unit_kind == "point"]
    assert "1" in point_values and "2" in point_values


def test_parse_decree_rtf_without_postanovlyayu_still_splits(tmp_path):
    content = (
        r"{\rtf1\ansi УКАЗ № 5 от 2 февраля 2026 г.\par "
        r"1. Первая позиция постановляющего характера.\par "
        r"2. Вторая позиция постановляющего характера.\par}"
    )
    file_path = tmp_path / "ukaz5.rtf"
    file_path.write_text(content, encoding="utf-8")
    units, doc_metadata = parse_decree_rtf(file_path, _decree_profile(), FakeSettings())
    assert doc_metadata.get("decree_number") == "5"
    assert any(u.unit_kind == "point" for u in units)


def test_parser_routes_decree_rtf_to_structured_path(tmp_path):
    file_path = tmp_path / "ukaz.rtf"
    file_path.write_text(_DECREE_RTF, encoding="utf-8")
    parser = LangchainDocumentParser(domain_registry=_RegistryStub(), domain_settings=FakeSettings())
    docs = parser.parse(file_path)
    # One RawDocument per structural unit, header metadata on each
    assert len(docs) > 1
    assert all(d.metadata.get("decree_number") == "123" for d in docs)
    assert any(d.metadata.get("unit_kind") == "point" for d in docs)

    # Splitter passes pre-split units through without re-splitting
    splitter = LangchainDocumentSplitter(domain_registry=_RegistryStub(), domain_settings=FakeSettings())
    chunks = splitter.split(docs, domain="decree")
    assert len(chunks) == len(docs)


def test_plain_rtf_uses_flat_path(tmp_path):
    file_path = tmp_path / "letter.rtf"
    file_path.write_text(_PLAIN_RTF, encoding="utf-8")
    parser = LangchainDocumentParser(domain_registry=_RegistryStub(), domain_settings=FakeSettings())
    docs = parser.parse(file_path)
    assert len(docs) == 1
    assert "unit_kind" not in docs[0].metadata


class _RegistryStub:
    """Minimal registry stub: only 'decree' is known."""

    def get(self, key: str):
        if key != "decree":
            raise KeyError(key)
        return _decree_profile()
