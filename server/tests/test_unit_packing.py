"""Structural packing preserves scope, source spans and atomic boundaries."""

import re
from types import SimpleNamespace

from langchain.schema import Document
import pytest

from domain.ingestion.content_splitter import BoundaryContext, SplitUnit, split_by_content
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.profiles.patterns import SENTENCE_RE
from domain.ingestion.profiles.types import BoundaryLevel
from domain.ingestion.unit_packing import pack_units
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.ingestion.splitting import split_documents
from infrastructure.ml.ingestion.structural_metadata import unit_metadata
from infrastructure.ml.ingestion.text_chunks import text_document_chunks


RANK = {"article": 0, "point": 1, "sentence": 2}


def test_sentences_pack_until_budget_and_keep_exact_repeated_source() -> None:
    text = "Повтор.  Повтор.\nПовтор."
    units = split_by_content(text, LegalDomainProfile().content_boundaries(), 8)
    groups = pack_units(units, 18, len, source=text, rank=RANK, scope_boundaries=frozenset({"article"}))
    assert [group.content for group in groups] == ["Повтор.  Повтор.", "Повтор."]
    assert all(text[group.start : group.end] == group.content for group in groups)
    assert groups[-1].start == text.rfind("Повтор.")


def test_distinct_articles_never_pack() -> None:
    text = "Статья 1. Один\n1. Первое правило.\nСтатья 2. Два\n1. Второе правило."
    units = split_by_content(text, LegalDomainProfile().content_boundaries(), 40)
    groups = pack_units(units, 1000, len, rank=RANK, source=text, scope_boundaries=frozenset({"article"}))
    assert len(groups) == 2
    assert all(
        not ("Первое правило" in group.content and "Второе правило" in group.content) for group in groups
    )


def test_fitting_point_is_atomic_and_numbers_are_collected() -> None:
    text = "1. Первое правило.\n2. Второе правило."
    units = split_by_content(text, LegalDomainProfile().content_boundaries(), 100)
    groups = pack_units(units, 100, len, source=text, rank=RANK, scope_boundaries=frozenset({"article"}))
    assert len(groups) == 1
    assert groups[0].point_numbers == ("1", "2")
    assert groups[0].boundary_values == ("1", "2")
    assert groups[0].boundary_value is None


def test_break_prefers_point_over_sentence_and_tail_rejoins_when_it_fits() -> None:
    units = [
        SplitUnit(None, text, kind)
        for text, kind in [("aaaa", "sentence"), ("bbbb", "point"), ("cccc", "sentence"), ("dd", "sentence")]
    ]
    groups = pack_units(units, 14, len, rank=RANK, min_size=5)
    assert [group.content for group in groups] == ["aaaa", "bbbb\ncccc\ndd"]
    assert all(len(group.content) <= 14 for group in groups)


def test_parent_scope_is_a_hard_barrier() -> None:
    units = [
        SplitUnit(None, "text", "sentence", parents=(BoundaryContext("point", "point", number, "intro"),))
        for number in ("1", "2")
    ]
    assert len(pack_units(units, 100, len)) == 2


def test_heading_only_intro_is_context_but_meaningful_intro_survives() -> None:
    text = (
        "Глава 1\nОБЩИЕ ПОЛОЖЕНИЯ\nСтатья 1. Термины\n"
        "Настоящий Закон регулирует отношения.\n1. Первое правило.\n2. Второе правило."
    )
    units = split_by_content(text, LegalDomainProfile().content_boundaries(), 60)
    assert any("Настоящий Закон регулирует" in unit.content for unit in units)
    assert not any(unit.content == "Глава 1\nОБЩИЕ ПОЛОЖЕНИЯ" for unit in units)
    assert any("Глава 1" in parent.content for unit in units for parent in unit.parents)
    assert all(text[unit.start : unit.end] == unit.content for unit in units)


def test_legal_letter_subpoint_has_point_parent_and_numeric_subpoints_accept_both_forms() -> None:
    text = (
        "Статья 1. Правила\n1. Условия:\nа) первое условие;\nб) второе условие.\n"
        "2. Новый пункт:\n2.1 Первое.\n2.2. Второе."
    )
    units = split_by_content(text, LegalDomainProfile().content_boundaries(), 35)
    letters = [unit for unit in units if unit.unit_kind == "subpoint"]
    assert letters
    assert any(parent.unit_kind == "point" and parent.boundary_value == "1" for parent in letters[0].parents)
    numeric = [unit.boundary_value for unit in units if unit.unit_kind == "subpoint_num"]
    assert numeric == ["2.1", "2.2"]


@pytest.mark.parametrize(
    "text",
    [
        "Раздел 1\nГлава 1\nСтатья 1. Термины\n1. " + "правило " * 30,
        "Глава 1\nРаздел 1\nСтатья 1. Термины\n1. " + "правило " * 30,
    ],
)
def test_section_chapter_order_follows_document(text: str) -> None:
    units = split_by_content(text, LegalDomainProfile().content_boundaries(), 60)
    kinds = [parent.unit_kind for parent in units[-1].parents]
    expected = ["section", "chapter"] if text.startswith("Раздел") else ["chapter", "section"]
    assert kinds[:2] == expected


def test_pdf_wrapped_reference_is_not_a_point() -> None:
    text = "Статья 1. Правила\nСогласно статье\n1. настоящего Закона действует правило."
    units = split_by_content(text, LegalDomainProfile().content_boundaries(), 40)
    assert not any(unit.boundary_value == "1" and unit.unit_kind == "point" for unit in units)


def test_custom_measure_controls_budget() -> None:
    units = [SplitUnit(None, "one two", "sentence") for _ in range(3)]
    groups = pack_units(units, 4, lambda text: len(re.findall(r"\w+", text)))
    assert len(groups) == 2


def test_custom_profile_controls_heading_blocks_and_packing_barriers() -> None:
    levels = [
        BoundaryLevel("part", re.compile(r"^Part (\d+)", re.MULTILINE), packing_barrier=True),
        BoundaryLevel("rule", re.compile(r"^Rule (\d+)", re.MULTILINE)),
    ]
    text = "Part 1\nTITLE\nRule 1 Text.\nRule 2 Other text."
    units = split_by_content(text, levels, 30)
    assert [unit.unit_kind for unit in units] == ["rule", "rule"]
    assert all(unit.parents[0].content == "Part 1\nTITLE" for unit in units)
    whole_parts = [SplitUnit(f"part {number}", "text", "part") for number in (1, 2)]
    assert len(pack_units(whole_parts, 100, len, scope_boundaries=frozenset({"part"}))) == 2


def test_short_articles_inside_one_chapter_still_remain_separate() -> None:
    text = "Глава 1\nПравила\nСтатья 1. Первый\nТекст первой нормы.\nСтатья 2. Второй\nТекст второй нормы."
    units = split_by_content(text, LegalDomainProfile().content_boundaries(), 1000)
    groups = pack_units(units, 1000, len, source=text, rank=RANK, scope_boundaries=frozenset({"article"}))
    assert len(groups) == 2
    assert [group.boundary_value for group in groups] == ["1", "2"]


def test_legal_article_produces_two_or_three_chunks_instead_of_sentence_fragments() -> None:
    settings = DomainSettingsAdapter()
    profile = LegalDomainProfile(settings)
    values = {(default.key, profile.key): default.value for default in profile.config_defaults()}
    values["legal_chunk_max_tokens", profile.key] = "1000"
    values["legal_chunk_target_tokens", profile.key] = "1000"
    settings.replace_snapshot(values)
    sentence = "Работодатель обеспечивает работнику условия для выполнения обязанностей. "
    text = "Глава 1\nОБЩИЕ ПОЛОЖЕНИЯ\nСтатья 1. Термины\n1. " + sentence * 10 + "\n2. " + sentence * 13
    chunks = split_documents(
        [Document(page_content=text)],
        domain="legal",
        profile=profile,
        settings=settings,
        token_counter=SimpleNamespace(count=len),
    )
    assert 2 <= len(chunks) <= 3
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)


def test_packed_repeated_bodies_keep_their_exact_page_ranges() -> None:
    settings = DomainSettingsAdapter()
    profile = LegalDomainProfile(settings)
    values = {(default.key, profile.key): default.value for default in profile.config_defaults()}
    values["legal_chunk_max_tokens", profile.key] = "1000"
    values["legal_chunk_target_tokens", profile.key] = "1000"
    settings.replace_snapshot(values)
    article = "Статья 1. Правила\n1. Повторяющийся текст.\n"
    text = article + article
    chunks = split_documents(
        [
            Document(
                page_content=text,
                metadata={"_page_spans": [(0, len(article), 1), (len(article), len(text), 2)]},
            )
        ],
        domain="legal",
        profile=profile,
        settings=settings,
        token_counter=SimpleNamespace(count=len),
    )
    assert len(chunks) == 2
    assert [chunk.metadata["pages"] for chunk in chunks] == [[1], [2]]


def test_point_and_letter_numbers_reach_final_metadata() -> None:
    settings = DomainSettingsAdapter()
    profile = LegalDomainProfile(settings)
    values = {(default.key, profile.key): default.value for default in profile.config_defaults()}
    values["legal_chunk_max_tokens", profile.key] = "100"
    values["legal_chunk_target_tokens", profile.key] = "100"
    settings.replace_snapshot(values)
    text = "Статья 1. Правила\n1. Условия:\nа) " + "первое условие; " * 4 + "\nб) " + "второе условие. " * 4
    chunks = split_documents(
        [Document(page_content=text)],
        domain="legal",
        profile=profile,
        settings=settings,
        token_counter=SimpleNamespace(count=len),
    )
    assert all(chunk.metadata["point_number"] == "1" for chunk in chunks)
    assert {chunk.metadata["subpoint_number"] for chunk in chunks} == {"а", "б"}


def legal_chunks(text: str) -> list[Document]:
    settings = DomainSettingsAdapter()
    profile = LegalDomainProfile(settings)
    values = {(default.key, profile.key): default.value for default in profile.config_defaults()}
    values["legal_chunk_max_tokens", profile.key] = "1000"
    values["legal_chunk_target_tokens", profile.key] = "1000"
    settings.replace_snapshot(values)
    return split_documents(
        [Document(page_content=text)],
        domain="legal",
        profile=profile,
        settings=settings,
        token_counter=SimpleNamespace(count=len),
    )


def test_long_intro_in_another_article_does_not_reduce_packing_budget() -> None:
    article = "Статья 2. Правила\n" + "\n".join(
        f"{number}. Работник исполняет обязанности в установленном порядке." for number in range(1, 30)
    )
    preceding = "Статья 1. Условия\n" + "Вводное условие действует для этой статьи; " * 14
    preceding += "\n1. " + "Самостоятельное правило. " * 45 + "\n"
    alone = legal_chunks(article)
    combined = [
        chunk for chunk in legal_chunks(preceding + article) if chunk.metadata.get("article_number") == "2"
    ]
    assert [chunk.page_content for chunk in combined] == [chunk.page_content for chunk in alone]
    assert len(alone) == 2


def test_article_over_body_budget_splits_at_complete_points() -> None:
    points = [
        f"{number}. " + "Работник исполняет обязанности согласно установленным правилам. " * 2
        for number in range(1, 10)
    ]
    text = "Статья 3. Порядок\n" + "\n".join(points)
    assert 1000 < len(text) < 1500
    chunks = legal_chunks(text)
    assert len(chunks) == 2
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)
    assert [number for chunk in chunks for number in chunk.metadata["point_numbers"]] == [
        str(number) for number in range(1, 10)
    ]
    assert all(sum(point.strip() in chunk.page_content for chunk in chunks) == 1 for point in points)


def test_compact_context_allows_fitting_article_points_to_pack() -> None:
    points = [f"{number}. " + "а" * 103 + "." for number in range(1, 10)]
    text = "Статья 3. Порядок\n" + "\n".join(points)
    assert 980 < len(text) < 1000
    chunks = legal_chunks(text)
    assert len(chunks) == 1
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)
    assert chunks[0].metadata["article_number"] == "3"
    assert all(point in chunks[0].metadata["body_text"] for point in points)


def test_anonymous_paragraph_does_not_become_a_heading() -> None:
    text = "Первый абзац.\n\nВторой абзац."
    levels = [level for level in LegalDomainProfile().content_boundaries() if level.name == "paragraph"]
    units = split_by_content(text, levels, 1000)
    assert len(units) == 2

    for unit in units:
        metadata = unit_metadata(unit, LegalDomainProfile(), {})
        assert unit.heading is None
        assert "section" not in metadata
        assert "unit_heading_text" not in metadata


def test_article_heading_is_not_repeated_in_its_own_body() -> None:
    heading = "Статья 3. Порядок"
    chunks = legal_chunks(heading + "\n" + "Установленное правило применяется ко всем работникам. " * 30)
    assert len(chunks) > 1
    assert chunks[0].page_content.count(heading) == 1


def test_overflow_renderer_does_not_duplicate_the_first_heading() -> None:
    heading = "Статья 3. Порядок"
    doc = Document(
        page_content=heading + "\n" + "Установленное правило применяется ко всем работникам. " * 22,
        metadata={"section": "article 3.", "unit_heading_text": heading},
    )
    chunks = text_document_chunks(
        doc, 1000, 250, [level.pattern for level in LegalDomainProfile().content_boundaries()]
    )
    assert len(chunks) > 1
    assert all(chunk.page_content.count(heading) == 1 for chunk in chunks)
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)


@pytest.mark.parametrize(
    "sentence",
    [
        "В соответствии со ст. 5 и п. 3 т.е. по ст. 12 норма действует.",
        "(в ред. Закона Республики Беларусь от 08.11.2018 N 143-З)",
        "Документ утв. постановлением, подписал А. Лукашенко.",
    ],
)
def test_legal_abbreviations_do_not_end_a_sentence(sentence: str) -> None:
    assert SENTENCE_RE.split(sentence) == [sentence]


def test_real_sentence_ends_still_split() -> None:
    assert SENTENCE_RE.split("Первое правило. Второе правило! Третье? Да.") == [
        "Первое правило.",
        "Второе правило!",
        "Третье?",
        "Да.",
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Согласно статье 15. Далее текст.", ["Согласно статье 15.", "Далее текст."]),
        ("1. Первый пункт.\n2. Второй.", ["1. Первый пункт.", "2. Второй."]),
        ("Статья 1. Термины", ["Статья 1. Термины"]),
        ("Глава 12. Общие положения", ["Глава 12. Общие положения"]),
        ("1.2. Вложенный пункт.", ["1.2. Вложенный пункт."]),
    ],
)
def test_number_ends_sentence_unless_it_numbers_a_unit(text: str, expected: list[str]) -> None:
    assert SENTENCE_RE.split(text) == expected


def test_capitalized_chapter_is_a_boundary_and_not_a_standalone_chunk() -> None:
    text = (
        "ГЛАВА 1\nОБЩИЕ ПОЛОЖЕНИЯ\nСтатья 1. Термины\n1. "
        + "Первое правило действует. " * 8
        + "\nГЛАВА 2\nЭЛЕКТРОННЫЙ ДОКУМЕНТ\nСтатья 2. Форма\n1. "
        + "Второе правило действует. " * 8
    )
    units = split_by_content(text, LegalDomainProfile().content_boundaries(), 60)
    assert {
        parent.boundary_value for unit in units for parent in unit.parents if parent.unit_kind == "chapter"
    } == {
        "1",
        "2",
    }
    assert not any(unit.content.startswith("ГЛАВА") for unit in units)


def test_separator_line_does_not_become_a_chunk() -> None:
    settings = DomainSettingsAdapter()
    profile = LegalDomainProfile(settings)
    values = {(default.key, profile.key): default.value for default in profile.config_defaults()}
    values["legal_chunk_max_tokens", profile.key] = "1000"
    values["legal_chunk_target_tokens", profile.key] = "1000"
    settings.replace_snapshot(values)
    text = "Статья 1. Правила\n1. Первое правило действует.\n" + "-" * 66
    chunks = split_documents(
        [Document(page_content=text)],
        domain="legal",
        profile=profile,
        settings=settings,
        token_counter=SimpleNamespace(count=len),
    )
    assert chunks
    assert all(any(char.isalnum() for char in chunk.page_content) for chunk in chunks)
    assert not any(
        chunk.page_content.strip().endswith("-" * 66) and len(chunk.page_content) < 80 for chunk in chunks
    )
