"""Tests for domain coverage gaps — exceptions, rag_policy, classifier, config_parameter."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app"))

from domain.entities.config_parameter import ConfigParameter
from domain.exceptions import (
    AppException,
    AuthenticationError,
    BusinessRuleViolation,
    ClientException,
    DatabaseError,
    EntityNotFound,
    PermissionDeniedError,
    ServerException,
    ValidationError,
)
from domain.services.document_domain_classifier import classify_document_domain
from domain.services.rag_policy import (
    build_system_prompt,
    classify_query_domain,
    has_exact_reference,
)
from domain.value_objects.config_value_type import ConfigValueType
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth


# ===========================================================================
# AppException Tests
# ===========================================================================


class TestAppException:
    def test_str(self):
        exc = AppException("msg")
        assert str(exc) == "msg"

    def test_repr(self):
        exc = AppException("msg", errors={"k": "v"})
        assert repr(exc) == "AppException(message='msg', errors={'k': 'v'})"

    def test_repr_without_errors(self):
        exc = AppException("msg")
        assert repr(exc) == "AppException(message='msg', errors=None)"

    def test_as_dict_with_errors(self):
        exc = AppException("msg", errors={"detail": "x"})
        assert exc.as_dict() == {"message": "msg", "errors": {"detail": "x"}}

    def test_as_dict_without_errors(self):
        exc = AppException("msg")
        assert exc.as_dict() == {"message": "msg"}

    def test_as_dict_with_empty_errors(self):
        exc = AppException("msg", errors={})
        assert exc.as_dict() == {"message": "msg"}


# ===========================================================================
# ServerException Tests
# ===========================================================================


class TestServerException:
    def test_is_app_exception(self):
        assert issubclass(ServerException, AppException)

    def test_message(self):
        exc = ServerException("server error")
        assert exc.message == "server error"


# ===========================================================================
# DatabaseError Tests
# ===========================================================================


class TestDatabaseError:
    def test_default_message(self):
        exc = DatabaseError()
        assert exc.message == "Ошибка взаимодействия с БД"
        assert exc.errors is None

    def test_with_detail(self):
        exc = DatabaseError(detail="connection refused")
        assert exc.errors == {"detail": "connection refused"}

    def test_is_server_exception(self):
        assert issubclass(DatabaseError, ServerException)


# ===========================================================================
# ClientException Tests
# ===========================================================================


class TestClientException:
    def test_is_app_exception(self):
        assert issubclass(ClientException, AppException)


# ===========================================================================
# ValidationError Tests
# ===========================================================================


class TestValidationError:
    def test_default_message(self):
        exc = ValidationError()
        assert exc.message == "Ошибка валидации"
        assert exc.errors is None

    def test_custom_message(self):
        exc = ValidationError("bad input")
        assert exc.message == "bad input"

    def test_with_errors_dict(self):
        exc = ValidationError("err", errors={"field": "msg"})
        assert exc.errors == {"field": "msg"}

    def test_with_field_param(self):
        exc = ValidationError("bad value", field="email")
        assert exc.message == "Ошибка валидации"
        assert exc.errors == {"email": "bad value"}

    def test_field_param_ignored_when_errors_given(self):
        exc = ValidationError("err", errors={"x": "y"}, field="email")
        assert exc.errors == {"x": "y"}
        assert exc.message == "err"


# ===========================================================================
# EntityNotFound Tests
# ===========================================================================


class TestEntityNotFound:
    def test_message_and_errors(self):
        exc = EntityNotFound("User", 42)
        assert exc.message == "User with id=42 not found"
        assert exc.errors == {"user": "User with id=42 not found", "id": "42"}

    def test_string_identifier(self):
        exc = EntityNotFound("Document", "abc-123")
        assert "abc-123" in exc.message
        assert exc.errors["id"] == "abc-123"


# ===========================================================================
# AuthenticationError Tests
# ===========================================================================


class TestAuthenticationError:
    def test_default(self):
        exc = AuthenticationError()
        assert exc.message == "Не авторизован"
        assert exc.errors == {"detail": "Не авторизован"}

    def test_custom_detail(self):
        exc = AuthenticationError("token expired")
        assert exc.message == "token expired"
        assert exc.errors == {"detail": "token expired"}


# ===========================================================================
# PermissionDeniedError Tests
# ===========================================================================


class TestPermissionDeniedError:
    def test_no_permission(self):
        exc = PermissionDeniedError()
        assert exc.message == "Недостаточно прав"
        assert exc.errors is None

    def test_single_permission(self):
        exc = PermissionDeniedError("admin")
        assert exc.errors == {"required_permission": "admin"}

    def test_multiple_permissions(self):
        exc = PermissionDeniedError(["admin", "moderator"])
        assert exc.errors == {"required_any_of": ["admin", "moderator"]}


# ===========================================================================
# BusinessRuleViolation Tests
# ===========================================================================


class TestBusinessRuleViolation:
    def test_is_client_exception(self):
        assert issubclass(BusinessRuleViolation, ClientException)


# ===========================================================================
# ConfigParameter.validate() Tests
# ===========================================================================


class TestConfigParameterValidate:
    def test_bool_valid_true(self):
        p = ConfigParameter(key="k", value="1", value_type=ConfigValueType.BOOL, category="c")
        p.validate("true")

    def test_bool_valid_false(self):
        p = ConfigParameter(key="k", value="1", value_type=ConfigValueType.BOOL, category="c")
        p.validate("false")

    def test_bool_invalid(self):
        p = ConfigParameter(key="k", value="1", value_type=ConfigValueType.BOOL, category="c")
        with pytest.raises(ValidationError, match="must be boolean"):
            p.validate("maybe")

    def test_str_valid(self):
        p = ConfigParameter(
            key="k", value="a", value_type=ConfigValueType.STR, category="c", allowed_values=["a", "b"]
        )
        p.validate("a")

    def test_str_invalid_not_in_allowed(self):
        p = ConfigParameter(
            key="k", value="a", value_type=ConfigValueType.STR, category="c", allowed_values=["a", "b"]
        )
        with pytest.raises(ValidationError, match="must be one of"):
            p.validate("c")

    def test_str_no_allowed_values(self):
        p = ConfigParameter(key="k", value="x", value_type=ConfigValueType.STR, category="c")
        p.validate("anything")

    def test_int_valid(self):
        p = ConfigParameter(key="k", value="5", value_type=ConfigValueType.INT, category="c")
        p.validate("10")

    def test_int_invalid(self):
        p = ConfigParameter(key="k", value="5", value_type=ConfigValueType.INT, category="c")
        with pytest.raises(ValidationError, match="Invalid value"):
            p.validate("abc")

    def test_int_below_min(self):
        p = ConfigParameter(key="k", value="5", value_type=ConfigValueType.INT, category="c", min_value=0)
        with pytest.raises(ValidationError, match="must be >="):
            p.validate("-1")

    def test_int_above_max(self):
        p = ConfigParameter(key="k", value="5", value_type=ConfigValueType.INT, category="c", max_value=100)
        with pytest.raises(ValidationError, match="must be <="):
            p.validate("200")

    def test_int_within_bounds(self):
        p = ConfigParameter(
            key="k", value="5", value_type=ConfigValueType.INT, category="c", min_value=0, max_value=100
        )
        p.validate("50")

    def test_float_valid(self):
        p = ConfigParameter(key="k", value="1.0", value_type=ConfigValueType.FLOAT, category="c")
        p.validate("2.5")

    def test_float_invalid(self):
        p = ConfigParameter(key="k", value="1.0", value_type=ConfigValueType.FLOAT, category="c")
        with pytest.raises(ValidationError, match="Invalid value"):
            p.validate("not_a_float")

    def test_float_below_min(self):
        p = ConfigParameter(
            key="k", value="1.0", value_type=ConfigValueType.FLOAT, category="c", min_value=0.0
        )
        with pytest.raises(ValidationError, match="must be >="):
            p.validate("-0.5")

    def test_float_above_max(self):
        p = ConfigParameter(
            key="k", value="1.0", value_type=ConfigValueType.FLOAT, category="c", max_value=10.0
        )
        with pytest.raises(ValidationError, match="must be <="):
            p.validate("20.0")

    def test_unknown_type_returns(self):
        p = ConfigParameter(key="k", value="x", value_type="unknown", category="c")
        p.validate("anything")

    def test_int_no_bounds(self):
        p = ConfigParameter(key="k", value="5", value_type=ConfigValueType.INT, category="c")
        p.validate("999")

    def test_float_no_bounds(self):
        p = ConfigParameter(key="k", value="1.0", value_type=ConfigValueType.FLOAT, category="c")
        p.validate("999.9")


# ===========================================================================
# build_system_prompt Tests
# ===========================================================================


class TestBuildSystemPrompt:
    def test_narrow_prompt(self):
        prompt = build_system_prompt(Breadth.NARROW)
        assert "КРАТКО" in prompt
        assert "{context}" in prompt

    def test_broad_prompt(self):
        prompt = build_system_prompt(Breadth.BROAD)
        assert "РАЗВЁРНУТО" in prompt
        assert "{context}" in prompt

    def test_default_is_narrow(self):
        prompt = build_system_prompt()
        assert "КРАТКО" in prompt

    def test_legal_context_appended(self):
        prompt = build_system_prompt(Breadth.NARROW, has_legal_context=True)
        assert "ЮРИДИЧЕСКОГО КОНТЕКСТА" in prompt
        assert "статьи/пункта" in prompt

    def test_no_legal_context(self):
        prompt = build_system_prompt(Breadth.NARROW, has_legal_context=False)
        assert "ЮРИДИЧЕСКОГО КОНТЕКСТА" not in prompt


# ===========================================================================
# classify_query_domain Tests
# ===========================================================================


class TestClassifyQueryDomain:
    def test_legal_patterns(self):
        assert classify_query_domain("Вправе ли работник?") == DocDomain.LEGAL
        assert classify_query_domain("Обязан ли арендатор?") == DocDomain.LEGAL
        assert classify_query_domain("Подлежит ли регистрации?") == DocDomain.LEGAL
        assert classify_query_domain("Несёт ли ответственность?") == DocDomain.LEGAL
        assert classify_query_domain("Статья 15 ФЗ") == DocDomain.LEGAL
        assert classify_query_domain("Пункт 3 договора") == DocDomain.LEGAL
        assert classify_query_domain("В соответствии с законом") == DocDomain.LEGAL
        assert classify_query_domain("Согласно договору") == DocDomain.LEGAL
        assert classify_query_domain("Нарушение условий договора") == DocDomain.LEGAL

    def test_general_patterns(self):
        assert classify_query_domain("Какой пароль?") == DocDomain.GENERAL
        assert classify_query_domain("Где скачать программу?") == DocDomain.GENERAL
        assert classify_query_domain("") == DocDomain.GENERAL


# ===========================================================================
# has_exact_reference Tests
# ===========================================================================


class TestHasExactReference:
    def test_article_reference(self):
        assert has_exact_reference("Смотри статья 15") is True

    def test_paragraph_reference(self):
        assert has_exact_reference("Согласно п. 3.2") is True

    def test_section_reference(self):
        assert has_exact_reference("раздел 4 описано") is True

    def test_chapter_reference(self):
        assert has_exact_reference("Глава 2 этого документа") is True

    def test_st_reference(self):
        assert has_exact_reference("ст. 10 УК РФ") is True

    def test_no_reference(self):
        assert has_exact_reference("Общая информация") is False

    def test_empty_string(self):
        assert has_exact_reference("") is False


# ===========================================================================
# classify_document_domain Tests
# ===========================================================================


class TestClassifyDocumentDomain:
    def test_general_text(self):
        assert classify_document_domain("Простой текст без юридических терминов") == DocDomain.GENERAL

    def test_legal_text_high_density(self):
        text = "\nСтатья 1\n" * 10 + "\nГлава 2\n" * 10
        assert classify_document_domain(text, threshold=1.0) == DocDomain.LEGAL

    def test_short_text_with_markers(self):
        text = "Федеральный закон\nГК РФ\nНК РФ"
        assert classify_document_domain(text, threshold=1.0) == DocDomain.LEGAL

    def test_empty_text(self):
        assert classify_document_domain("") == DocDomain.GENERAL

    def test_custom_threshold(self):
        text = "Федеральный закон"
        assert classify_document_domain(text, threshold=100.0) == DocDomain.GENERAL
        assert classify_document_domain(text, threshold=0.1) == DocDomain.LEGAL

    def test_all_markers(self):
        text = (
            "\nСтатья 1\n"
            "\nГлава 2\n"
            "\nРаздел 3\n"
            "\nПункт 4\n"
            "\n1.2.\n"
            "Федеральный закон"
            "ГК РФ"
            "НК РФ"
            "настоящим договором"
            "стороны договорились"
        )
        assert classify_document_domain(text, threshold=0.5) == DocDomain.LEGAL
