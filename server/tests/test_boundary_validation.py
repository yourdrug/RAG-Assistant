"""Structural markers are accepted only within their actual numbering scope."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from domain.ingestion.boundary_validation import boundary_candidates, validate_candidates
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.profiles.patterns import POINT_RE, SUBPOINT_LETTER_RE, SUBPOINT_NUM_RE
from domain.ingestion.profiles.types import BoundaryLevel
from domain.ingestion.tree_packer import select_structural_units


def accepted(
    text: str,
    level: BoundaryLevel,
    *,
    parent_number: str | None = None,
    on_event: Callable[[str, int], None] | None = None,
) -> list[str | None]:
    return [
        item.value
        for item in validate_candidates(
            boundary_candidates(text, level),
            text,
            [level],
            parent_number=parent_number,
            on_event=on_event,
        )
    ]


def test_wrapped_numeric_reference_is_not_a_point() -> None:
    text = "В соответствии с пунктом\n1. настоящего Закона установлены правила."
    assert accepted(text, BoundaryLevel("point", POINT_RE)) == []


def test_article_reference_is_not_a_heading() -> None:
    profile = LegalDomainProfile()
    text = "Статья 12 настоящего Закона применяется.\nСтатья 13. Права работника"
    units = select_structural_units(text, profile.content_boundaries(), 1000)
    assert [unit.boundary_value for unit in units if unit.unit_kind == "article"] == ["13"]
    assert any("Статья 12" in unit.content for unit in units)


def test_nested_number_reset_is_text_until_a_new_parent() -> None:
    text = "1. Первое.\n2. Второе.\n3. Вложенный список:\n1. Вложенное правило."
    assert accepted(text, BoundaryLevel("point", POINT_RE)) == ["1", "2", "3"]
    article_text = "Статья 1. Один\n1. Первый пункт.\nСтатья 2. Два\n1. Другой пункт."
    units = select_structural_units(article_text, LegalDomainProfile().content_boundaries(), 20)
    points = [unit for unit in units if unit.boundary_value == "1" and unit.unit_kind == "point"]
    assert len(points) == 2
    assert points[0].parents != points[1].parents


def test_single_missing_number_is_reported_and_accepted() -> None:
    events = []
    text = "1. Первое.\n3. Третье.\n7. Ссылка."
    assert accepted(text, BoundaryLevel("point", POINT_RE), on_event=lambda *event: events.append(event)) == [
        "1",
        "3",
    ]
    assert [event[0] for event in events] == ["number_gap", "number_order"]


@pytest.mark.parametrize("parent, expected", [(None, []), ("1", ["1.1", "1.2"])])
def test_numeric_subpoints_require_their_open_parent(parent: str | None, expected: list[str]) -> None:
    text = "1.1. Первое.\n1.2. Второе."
    assert accepted(text, BoundaryLevel("subpoint_num", SUBPOINT_NUM_RE), parent_number=parent) == expected


def test_nested_subpoints_keep_independent_sequences() -> None:
    text = "1.1. Условия:\n1.1.1. Первое.\n1.1.2. Второе.\n1.2. Следующее."
    assert accepted(text, BoundaryLevel("subpoint_num", SUBPOINT_NUM_RE), parent_number="1") == [
        "1.1",
        "1.1.1",
        "1.1.2",
        "1.2",
    ]


def test_article_order_resets_in_a_new_chapter() -> None:
    text = "Глава 1\nСтатья 3. Первая\nСтатья 2. Ссылка\nГлава 2\nСтатья 1. Новая"
    units = select_structural_units(text, LegalDomainProfile().content_boundaries(), 1000)
    assert [unit.boundary_value for unit in units if unit.unit_kind == "article"] == ["3", "1"]
    assert any("Статья 2" in unit.content for unit in units)


def test_format_hint_can_confirm_a_numeric_heading_after_plain_text() -> None:
    text = "Вводная часть\n1. Новый пункт."
    level = BoundaryLevel("point", POINT_RE)
    candidates = boundary_candidates(text, level)
    assert validate_candidates(candidates, text, [level]) == []
    assert validate_candidates([replace(candidates[0], format_confirmed=True)], text, [level])


@pytest.mark.parametrize("heading", ["ГЛАВА 2", "Раздел 2", "Статья 2", "Приложение 2"])
def test_first_point_after_two_line_parent_heading_is_accepted(heading: str) -> None:
    text = f"2. Предыдущее правило.\n{heading}\nПОРЯДОК РЕАЛИЗАЦИИ\n3. Реализации подлежат товары."
    assert accepted(text, BoundaryLevel("point", POINT_RE)) == ["2", "3"]


@pytest.mark.parametrize(
    "title",
    [
        "ИНСТРУКЦИЯ\nО ПОРЯДКЕ МАРКИРОВКИ ОСТАТКОВ ТОВАРОВ",
        "СОСТАВ\nСВЕДЕНИЙ, ПРЕДСТАВЛЯЕМЫХ EDI-ПРОВАЙДЕРАМИ В ПК СПТ",
    ],
)
def test_first_point_after_uppercase_document_title_is_accepted(title: str) -> None:
    text = (
        f"2. Пункт постановления.\nПриложение\nк постановлению\n{title}\n1. Настоящая Инструкция определяет."
    )
    assert accepted(text, BoundaryLevel("point", POINT_RE)) == ["2", "1"]


def test_uppercase_title_does_not_restart_numbering_mid_list() -> None:
    text = "1. Первый.\n2. Второй.\nПОРЯДОК РАСЧЕТОВ\n5. Несогласованный номер."
    assert accepted(text, BoundaryLevel("point", POINT_RE)) == ["1", "2"]


@pytest.mark.parametrize("line", ["XML", "ЭТТН И ЭТН,", "в формате ЭТН"])
def test_short_or_wrapped_line_does_not_confirm_a_point(line: str) -> None:
    text = f"В соответствии с документом\n{line}\n3. настоящего постановления."
    assert accepted(text, BoundaryLevel("point", POINT_RE)) == []


def test_chapter_heading_does_not_confirm_a_later_wrapped_reference() -> None:
    text = "ГЛАВА 2\nПОРЯДОК РЕАЛИЗАЦИИ\nВ соответствии с пунктом\n3. настоящего постановления."
    assert accepted(text, BoundaryLevel("point", POINT_RE)) == []


def test_point_after_chapter_opens_parent_scope_for_numeric_subpoints() -> None:
    text = (
        "8. Предыдущее правило.\nГЛАВА 2\nПОРЯДОК РЕАЛИЗАЦИИ\n"
        "9. Применяются условия:\n9.1. Первое условие.\n9.2. Второе условие."
    )
    units = select_structural_units(text, DecreeDomainProfile().content_boundaries(), 1000)
    subpoints = [unit for unit in units if unit.unit_kind == "subpoint_num"]
    assert [unit.boundary_value for unit in subpoints] == ["9.1", "9.2"]
    assert all(unit.parents[-1].boundary_value == "9" for unit in subpoints)
    assert all("9." not in unit.content for unit in units if unit.boundary_value == "8")


@pytest.mark.parametrize(
    "note",
    [
        "(в ред. постановления от 01.01.2020)",
        "(часть первая в ред. постановления от 01.01.2020)",
        "(пункты 3.5 и 3.6 в ред. постановления от 01.01.2020)",
        "(подпункты 3.5 и 3.6 в ред. постановления от 01.01.2020)",
        "(абзацы второй и третий исключены постановлением от 01.01.2020)",
        "(пп. 3.6 введен постановлением от 01.01.2020)",
        "(сноска<***> введена постановлением от 01.01.2020)",
        "(преамбула в ред. постановления от 01.01.2020)",
    ],
)
def test_editorial_note_does_not_hide_next_point(note: str) -> None:
    assert accepted(f"{note}\n1. Новое правило.", BoundaryLevel("point", POINT_RE)) == ["1"]
    assert accepted(
        f"3.6. Первое правило.\n{note}\n3.7. Новое правило.",
        BoundaryLevel("subpoint_num", SUBPOINT_NUM_RE),
        parent_number="3",
    ) == ["3.6", "3.7"]


@pytest.mark.parametrize("title", ["«Термины» и определения", "(исключена)", "общие положения"])
def test_article_titles_do_not_require_an_uppercase_initial(title: str) -> None:
    text = f"Статья 1. {title}\nСтатья 2. Следующая статья"
    units = select_structural_units(text, LegalDomainProfile().content_boundaries(), 1000)
    assert [unit.boundary_value for unit in units if unit.unit_kind == "article"] == ["1", "2"]


def test_inserted_article_keeps_its_full_number_and_order() -> None:
    text = (
        "Статья 26. Сертификат\nСтатья 26-1. Атрибутный сертификат\n"
        "Статья 26. Ссылка назад\nСтатья 27. Следующая статья"
    )
    units = select_structural_units(text, LegalDomainProfile().content_boundaries(), 1000)
    articles = [unit for unit in units if unit.unit_kind == "article"]
    assert [unit.boundary_value for unit in articles] == ["26", "26-1", "27"]
    assert "Статья 26-1" not in articles[0].content
    assert "Статья 26. Ссылка назад" in articles[1].content


def test_numbering_recovers_after_a_confirmed_forward_gap() -> None:
    events = []
    text = "1. Первое.\n2. Второе.\n3. Третье.\n6. Шестое.\n7. Седьмое.\n8. Восьмое."
    assert accepted(text, BoundaryLevel("point", POINT_RE), on_event=lambda *event: events.append(event)) == [
        "1",
        "2",
        "3",
        "6",
        "7",
        "8",
    ]
    assert events == [("number_gap", text.index("6."))]


def test_subpoint_numbering_recovers_within_its_open_parent() -> None:
    text = "3.1. Первое.\n3.4. Четвёртое.\n3.5. Пятое."
    assert accepted(text, BoundaryLevel("subpoint_num", SUBPOINT_NUM_RE), parent_number="3") == [
        "3.1",
        "3.4",
        "3.5",
    ]


def test_subpoint_recovery_can_find_confirming_sibling_after_descendants() -> None:
    text = "3.1. Первое.\n3.4. Условия:\n3.4.1. Вложенное условие.\n3.5. Пятое."
    assert accepted(text, BoundaryLevel("subpoint_num", SUBPOINT_NUM_RE), parent_number="3") == [
        "3.1",
        "3.4",
        "3.4.1",
        "3.5",
    ]


@pytest.mark.parametrize(
    "tail",
    [
        "6. Одиночная ссылка.",
        "6. Ссылка.\n8. Несогласованный номер.",
        "6. В соответствии с пунктом\n7. настоящего Закона.",
        "6. Ссылка.\nГЛАВА 2\n7. Другой scope.",
    ],
)
def test_forward_gap_needs_confirmation_in_the_same_scope(tail: str) -> None:
    text = "1. Первый пункт.\n2. Второй пункт.\n" + tail
    values = accepted(text, BoundaryLevel("point", POINT_RE))
    assert "6" not in values


def test_unconfirmed_point_before_a_second_gap_stays_text() -> None:
    # 6 has no successor 7, so it remains text; 9 is confirmed by 10.
    text = "1. Первый.\n2. Второй.\n3. Третий.\n6. Шестой.\n9. Девятый.\n10. Десятый."
    assert accepted(text, BoundaryLevel("point", POINT_RE)) == ["1", "2", "3", "9", "10"]


def test_nested_numbering_run_does_not_reset_parent_sequence() -> None:
    text = "1. Первое.\n2. Второе.\n3. Условия:\n1. Вложенное.\n2. Вложенное.\n4. Четвёртое."
    assert accepted(text, BoundaryLevel("point", POINT_RE)) == ["1", "2", "3", "4"]


@pytest.mark.parametrize("letters", ["дежзикл", "деёжзийкл"])
def test_letter_numbering_allows_optional_yo_and_short_i_without_gaps(letters: str) -> None:
    events = []
    text = "\n".join(f"{letter}) Условие." for letter in letters)
    assert accepted(
        text, BoundaryLevel("subpoint", SUBPOINT_LETTER_RE), on_event=lambda *event: events.append(event)
    ) == list(letters)
    assert events == []
