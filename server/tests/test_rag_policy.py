"""Tests for domain/services/rag_policy.py — pure business logic for question classification.

Functions tested:
  - classify_question_breadth
  - needs_decomposition
  - should_enumerate_cases
  - is_out_of_domain
  - classify_query_domain
  - has_exact_reference
  - build_system_prompt

All tests are pure unit tests with no infrastructure dependencies.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app"))

from domain.services.rag_policy import (
    build_system_prompt,
    classify_question_breadth,
    classify_query_domain,
    has_exact_reference,
    is_out_of_domain,
    needs_decomposition,
    should_enumerate_cases,
)
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth


# ===========================================================================
# classify_question_breadth
# ===========================================================================


class TestClassifyQuestionBreadth:
    """Business rule: classify questions as narrow or broad based on heuristics."""

    # --- Narrow overrides (high priority) ---

    def test_how_to_check_is_narrow(self):
        assert classify_question_breadth("Как проверить статус маркировки?") == Breadth.NARROW

    def test_where_to_find_is_narrow(self):
        assert classify_question_breadth("Где скачать форму заявления?") == Breadth.NARROW

    def test_what_is_narrow(self):
        assert classify_question_breadth("Что такое ЭТТН?") == Breadth.NARROW

    def test_which_exceptions_is_narrow(self):
        assert classify_question_breadth("Какие исключения из правила?") == Breadth.NARROW

    def test_why_is_narrow(self):
        assert classify_question_breadth("Почему отклонили заявку?") == Breadth.NARROW

    def test_can_i_is_narrow(self):
        assert classify_question_breadth("Можно ли подать заявку онлайн?") == Breadth.NARROW

    # --- Broad patterns ---

    def test_detailed_is_broad(self):
        assert classify_question_breadth("Подробно опиши процесс маркировки") == Breadth.BROAD

    def test_explain_everything_is_broad(self):
        assert classify_question_breadth("Объясни всё про штрафы") == Breadth.BROAD

    def test_tell_about_is_broad(self):
        assert classify_question_breadth("Расскажи про систему маркировки") == Breadth.BROAD

    def test_how_works_is_broad(self):
        assert classify_question_breadth("Как работает процесс оформления?") == Breadth.BROAD

    def test_order_of_is_broad(self):
        assert classify_question_breadth("Порядок получения кодов маркировки") == Breadth.BROAD

    def test_all_conditions_is_broad(self):
        assert classify_question_breadth("Какие условия для получения?") == Breadth.BROAD

    def test_list_is_broad(self):
        assert classify_question_breadth("Список документов для подачи") == Breadth.BROAD

    # --- Default ---

    def test_short_question_defaults_to_narrow(self):
        assert classify_question_breadth("Статус заявки?") == Breadth.NARROW

    def test_empty_string_is_narrow(self):
        assert classify_question_breadth("") == Breadth.NARROW

    # --- Override priority ---

    def test_narrow_override_wins_over_broad_pattern(self):
        # "Как проверить" is narrow override, even though it could match broad patterns
        q = "Как проверить статус заявки подробно"
        assert classify_question_breadth(q) == Breadth.NARROW


# ===========================================================================
# needs_decomposition
# ===========================================================================


class TestNeedsDecomposition:
    """Business rule: detect compound questions that benefit from decomposition."""

    def test_compare_x_and_y(self):
        assert needs_decomposition("Сравни маркировку и сертификацию") is True

    def test_tell_about_x_and_y(self):
        assert needs_decomposition("Расскажи про штрафы и про порядок обжалования") is True

    def test_how_x_and_y(self):
        assert needs_decomposition("Как оформить заявку и как получить код?") is True

    def test_simple_question_no_decomposition(self):
        assert needs_decomposition("Какой статус заявки?") is False

    def test_empty_string(self):
        assert needs_decomposition("") is False

    def test_single_topic_no_decomposition(self):
        assert needs_decomposition("Порядок получения кодов маркировки") is False


# ===========================================================================
# should_enumerate_cases
# ===========================================================================


class TestShouldEnumerateCases:
    """Business rule: detect when context has conditional/exclusive rules."""

    def test_single_chunk_not_enough(self):
        assert should_enumerate_cases("вопрос", ["один чанк"]) is False

    def test_empty_context(self):
        assert should_enumerate_cases("вопрос", []) is False

    def test_two_chunks_with_conditions(self):
        texts = [
            "В случае нарушения штраф 10000 руб.",
            "При условии соответствия выдаётся сертификат",
        ]
        assert should_enumerate_cases("вопрос", texts) is True

    def test_two_chunks_without_conditions(self):
        texts = [
            "Маркировка товаров обязательна",
            "Коды выдаются оператором",
        ]
        assert should_enumerate_cases("вопрос", texts) is False

    def test_one_chunk_with_conditions_not_enough(self):
        texts = ["В случае нарушения штраф 10000 руб."]
        assert should_enumerate_cases("вопрос", texts) is False

    def test_percentage_in_context(self):
        texts = ["Штраф составляет 5% от суммы", "НДС 20% от стоимости"]
        assert should_enumerate_cases("вопрос", texts) is True

    def test_exception_marker(self):
        texts = [
            "За исключением случаев force majeure",
            "При условии соответствия выдаётся сертификат",
        ]
        assert should_enumerate_cases("вопрос", texts) is True

    def test_none_entries_in_context(self):
        texts = [None, "В случае нарушения штраф", "При условии оплаты"]
        assert should_enumerate_cases("вопрос", texts) is True

    def test_empty_string_entries(self):
        texts = ["", "В случае нарушения штраф", "При условии оплаты"]
        assert should_enumerate_cases("вопрос", texts) is True


# ===========================================================================
# is_out_of_domain
# ===========================================================================


class TestIsOutOfDomain:
    """Business rule: filter out non-corporate questions before retrieval."""

    # --- Corporate questions (in domain) ---

    def test_markeeringa_question_is_in_domain(self):
        assert is_out_of_domain("Как оформить маркировку товаров?") is False

    def test_shtrafy_question_is_in_domain(self):
        assert is_out_of_domain("Какие штрафы за нарушение?") is False

    def test_poryadok_question_is_in_domain(self):
        assert is_out_of_domain("Порядок получения кодов") is False

    # --- Out of domain ---

    def test_codeforces_is_out_of_domain(self):
        assert is_out_of_domain("Реши задачу Codeforces 1234A") is True

    def test_leetcode_is_out_of_domain(self):
        assert is_out_of_domain("Algorithm for LeetCode problem") is True

    def test_recipe_is_out_of_domain(self):
        assert is_out_of_domain("рецепта готовки борща") is True

    def test_weather_is_out_of_domain(self):
        assert is_out_of_domain("Прогноз погоды на завтра") is True

    def test_horoscope_is_out_of_domain(self):
        assert is_out_of_domain("Гороскоп на сегодня") is True

    def test_newton_laws_is_out_of_domain(self):
        assert is_out_of_domain("Закона Ньютона в физике") is True

    def test_tutorial_is_out_of_domain(self):
        assert is_out_of_domain("Tutorial: как написать скрипт на Python") is True

    def test_quantum_mechanics_is_out_of_domain(self):
        assert is_out_of_domain("Квантовая механика для начинающих") is True

    def test_algorithm_is_out_of_domain(self):
        assert is_out_of_domain("Алгоритм быстрой сортировки") is True

    # --- Edge cases ---

    def test_empty_string_is_in_domain(self):
        assert is_out_of_domain("") is False

    def test_case_insensitive(self):
        assert is_out_of_domain("CODEFORCES задача") is True


# ===========================================================================
# classify_query_domain
# ===========================================================================


class TestClassifyQueryDomain:
    """Business rule: classify query as legal or general."""

    def test_legal_question_with_article(self):
        assert classify_query_domain("Статья 15 ФЗ-XXX говорит о") == DocDomain.LEGAL

    def test_legal_question_with_clause(self):
        assert classify_query_domain("Пункт 3.2 договора") == DocDomain.LEGAL

    def test_legal_question_with_accordance(self):
        assert classify_query_domain("В соответствии с законом") == DocDomain.LEGAL

    def test_general_question(self):
        assert classify_query_domain("Какой статус заявки?") == DocDomain.GENERAL

    def test_general_question_empty(self):
        assert classify_query_domain("") == DocDomain.GENERAL


# ===========================================================================
# has_exact_reference
# ===========================================================================


class TestHasExactReference:
    """Business rule: detect structural references in questions."""

    def test_article_reference(self):
        assert has_exact_reference("Что говорит статья 15?") is True

    def test_clause_reference(self):
        assert has_exact_reference("Пункт 3.2 договора") is True

    def test_section_reference(self):
        assert has_exact_reference("Раздел 2 документа") is True

    def test_chapter_reference(self):
        assert has_exact_reference("Глава 1 описывает") is True

    def test_no_reference(self):
        assert has_exact_reference("Общая информация о маркировке") is False

    def test_empty_string(self):
        assert has_exact_reference("") is False

    def test_abbreviated_forms(self):
        assert has_exact_reference("ст. 15 ФЗ") is True
        assert has_exact_reference("п. 3.2") is True


# ===========================================================================
# build_system_prompt
# ===========================================================================


class TestBuildSystemPrompt:
    """Business rule: prompt construction depends on breadth and context."""

    def test_narrow_prompt_brief(self):
        prompt = build_system_prompt(breadth=Breadth.NARROW)
        assert "КРАТКО" in prompt

    def test_broad_prompt_detailed(self):
        prompt = build_system_prompt(breadth=Breadth.BROAD, enumerate_cases=True)
        assert "подпунктам" in prompt.lower() or "ПОДПУНКТАМ" in prompt

    def test_broad_prompt_without_enumerate(self):
        prompt = build_system_prompt(breadth=Breadth.BROAD, enumerate_cases=False)
        assert "conditional_rules_expansion" not in prompt

    def test_domain_addendum_appears_in_prompt(self):
        prompt = build_system_prompt(
            breadth=Breadth.NARROW,
            domain_addendum="Обязательно указывай номер статьи/пункта.",
        )
        assert "domain_specific_rules" in prompt
        assert "Обязательно указывай номер статьи/пункта." in prompt

    def test_no_domain_addendum_no_domain_block(self):
        prompt = build_system_prompt(breadth=Breadth.NARROW)
        assert "domain_specific_rules" not in prompt

    def test_domain_addendum_overrides_generic_rules(self):
        prompt = build_system_prompt(
            breadth=Breadth.NARROW,
            domain_addendum="Дополнительное правило домена",
        )
        assert "Дополнительное правило домена" in prompt
        assert "domain_specific_rules" in prompt

    def test_not_found_phrase_in_prompt(self):
        prompt = build_system_prompt()
        assert "Информация не найдена в документах" in prompt
