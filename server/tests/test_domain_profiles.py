"""Tests for document profiles and their independent answer policies."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.profiles.protocol import (
    DocumentProfile,
)
from infrastructure.domain_profile.answer_policy import policy_for
from infrastructure.ml.rag.rag_prompts import domain_prompt_addendum


if TYPE_CHECKING:
    from domain.rag.context import RagContext


class FakeSettings:
    """In-memory DomainSettingsPort matching seeded defaults."""

    def get(self, key: str, domain_key: str = "") -> str:
        _ = domain_key
        defaults: dict[str, str] = {
            "max_unit_chars": "1200",
            "fingerprint_min_points": "1",
            "fingerprint_min_articles": "2",
            "classification_threshold": "2.0",
            "effective_date_auto_trust_threshold": "0.85",
        }
        return defaults.get(key, "0")


class TestProtocolConformance:
    def test_general_satisfies_protocol(self) -> None:
        p = GeneralDomainProfile()
        assert isinstance(p, DocumentProfile)

    def test_legal_satisfies_protocol(self) -> None:
        p = LegalDomainProfile()
        assert isinstance(p, DocumentProfile)

    def test_decree_satisfies_protocol(self) -> None:
        p = DecreeDomainProfile()
        assert isinstance(p, DocumentProfile)

    def test_general_is_not_versioned(self) -> None:
        assert GeneralDomainProfile().is_versioned is False

    def test_legal_is_versioned(self) -> None:
        assert LegalDomainProfile().is_versioned is True

    def test_decree_is_versioned(self) -> None:
        assert DecreeDomainProfile().is_versioned is True


class TestGeneralProfile:
    def test_config_defaults_empty(self) -> None:
        assert GeneralDomainProfile().config_defaults() == []

    def test_fingerprint_always_false(self) -> None:
        assert GeneralDomainProfile().structural_fingerprint("any text") is False

    def test_classify_score_zero(self) -> None:
        assert GeneralDomainProfile().classify_score("any text") == 0.0

    def test_content_boundaries_empty(self) -> None:
        assert GeneralDomainProfile().content_boundaries() == []

    def test_extract_references_empty(self) -> None:
        assert GeneralDomainProfile().extract_references("any text") == []

    def test_prompt_addendum_none(self) -> None:
        assert policy_for("general").prompt_addendum() is None


class TestDecreeProfile:
    def test_fingerprint_detects_decree(self) -> None:
        text = """
        УКАЗ ПРЕЗИДЕНТА
        О بعضых мерах

        ПОСТАНОВЛЯЮ:

        1. Принять предложенные меры.
        2. Контроль возложить.
        """
        assert DecreeDomainProfile(settings=FakeSettings()).structural_fingerprint(text) is True

    def test_fingerprint_rejects_non_decree(self) -> None:
        text = "Это обычный документ без структуры указа."
        assert DecreeDomainProfile(settings=FakeSettings()).structural_fingerprint(text) is False

    def test_classify_score_high_for_decree(self) -> None:
        text = "УКАЗ\nПОСТАНОВЛЯЮ:\n1. Принять.\n2. Контроль.\n3. Организовать."
        score = DecreeDomainProfile().classify_score(text)
        assert score > 3.0

    def test_classify_score_low_for_general(self) -> None:
        text = "Обычный текст без маркеров указа."
        score = DecreeDomainProfile().classify_score(text)
        assert score == 0.0

    def test_content_boundaries_has_point_subpoint_sentence(self) -> None:
        boundaries = DecreeDomainProfile().content_boundaries()
        names = [b.name for b in boundaries]
        assert "point" in names
        assert "subpoint" in names
        assert "sentence" in names

    def test_extract_references_decree_number(self) -> None:
        text = "Указ Президента №123 от 01 января 2026 г."
        refs = DecreeDomainProfile().extract_references(text)
        kinds = [r.kind for r in refs]
        assert "decree_number" in kinds

    def test_extract_references_points(self) -> None:
        text = "1. Первый пункт.\n2. Второй пункт.\n3. Третий пункт."
        refs = DecreeDomainProfile().extract_references(text)
        point_refs = [r for r in refs if r.kind == "point"]
        assert len(point_refs) == 3


class TestLegalProfile:
    def test_fingerprint_detects_legal(self) -> None:
        text = """
        Глава 1. Общие положения

        Статья 1. Основные понятия

        В настоящем законе используются следующие понятия.

        Статья 2. Предмет регулирования

        Настоящий закон регулирует.
        """
        assert LegalDomainProfile(settings=FakeSettings()).structural_fingerprint(text) is True

    def test_fingerprint_rejects_short_text(self) -> None:
        text = "Статья 1."
        assert LegalDomainProfile(settings=FakeSettings()).structural_fingerprint(text) is False

    def test_classify_score_legal_text(self) -> None:
        text = "Федеральный закон\nСтатья 1.\nСтатья 2.\nСтатья 3."
        score = LegalDomainProfile().classify_score(text)
        assert score > 1.0

    def test_content_boundaries_hierarchy(self) -> None:
        boundaries = LegalDomainProfile().content_boundaries()
        names = [b.name for b in boundaries]
        assert "chapter" in names
        assert "article" in names
        assert "sentence" in names

    def test_extract_references_articles(self) -> None:
        text = "Согласно ст. 15 ФЗ-123. Также ст. 20."
        refs = LegalDomainProfile().extract_references(text)
        article_refs = [r for r in refs if r.kind == "article"]
        assert len(article_refs) >= 1

    def test_prompt_addendum_contains_rules(self) -> None:
        addendum = policy_for("legal").prompt_addendum()
        assert addendum is not None
        assert "статьи" in addendum.lower() or "пункта" in addendum.lower()

    def test_prompt_addendum_with_as_of_date(self) -> None:
        addendum = policy_for("legal").prompt_addendum(as_of_date=date(2026, 1, 1))
        assert addendum is not None
        assert "01.01.2026" in addendum


# ---------------------------------------------------------------------------
# Effective date fallback
# ---------------------------------------------------------------------------


class TestEffectiveDateFallback:
    def test_legal_fallback_to_signing_date(self) -> None:
        """When no 'вступает в силу' pattern, falls back to signing date."""
        profile = LegalDomainProfile()
        text = "Федеральный закон от 15 марта 2026 года о чём-то.\nСтатья 1. Общие положения.\n"
        result = profile.extract_effective_date(text)
        assert result is not None
        assert result.signing_date == date(2026, 3, 15)
        assert result.confidence == 0.3  # fallback confidence

    def test_legal_explicit_effective_date(self) -> None:
        """'вступает в силу со дня' gets high confidence."""
        profile = LegalDomainProfile()
        text = "Федеральный закон вступает в силу со дня 1 июня 2026 года.\nСтатья 1. Общие положения.\n"
        result = profile.extract_effective_date(text)
        assert result is not None
        assert result.effective_from == date(2026, 6, 1)
        assert result.confidence >= 0.7

    def test_decree_fallback_to_signing_date(self) -> None:
        """Decree without 'вступает в силу' falls back to signing date."""
        profile = DecreeDomainProfile()
        text = "УКАЗ от 10 января 2026 г.\nПОСТАНОВЛЯЮ:\n1. Пункт.\n"
        result = profile.extract_effective_date(text)
        assert result is not None
        assert result.signing_date == date(2026, 1, 10)
        assert result.confidence == 0.3

    def test_no_date_returns_none(self) -> None:
        """Text with no date at all returns None."""
        profile = LegalDomainProfile()
        result = profile.extract_effective_date("Статья 1. Общие положения без дат.\n")
        assert result is None


# ---------------------------------------------------------------------------
# Decree fingerprint variations
# ---------------------------------------------------------------------------


class TestDecreeFingerprintVariations:
    def test_fingerprint_with_postanovlyaet(self) -> None:
        """ПОСТАНОВЛЯЕТ (not ПОСТАНОВЛЯЮ) should also fire fingerprint."""
        profile = DecreeDomainProfile(settings=FakeSettings())
        text = "УКАЗ ПРЕЗИДЕНТА\nПОСТАНОВЛЯЕТ:\n1. Пункт первый.\n"
        assert profile.structural_fingerprint(text) is True

    def test_fingerprint_requires_all_three_elements(self) -> None:
        """Missing any of УКАЗ, ПОСТАНОВЛЯЮ, or points → no fingerprint."""
        profile = DecreeDomainProfile(settings=FakeSettings())
        # Has УКАЗ and ПОСТАНОВЛЯЮ but no numbered points
        text = "УКАЗ\nПОСТАНОВЛЯЮ что-то.\n"
        assert profile.structural_fingerprint(text) is False

    def test_fingerprint_ignores_lowercase_ukaz(self) -> None:
        """'указ' (lowercase) should not fire fingerprint."""
        profile = DecreeDomainProfile(settings=FakeSettings())
        text = "указ президента\nПОСТАНОВЛЯЮ:\n1. Пункт.\n"
        assert profile.structural_fingerprint(text) is False


# ---------------------------------------------------------------------------
# Legal profile edge cases
# ---------------------------------------------------------------------------


class TestLegalProfileEdgeCases:
    def test_fingerprint_requires_chapter_and_articles(self) -> None:
        """Legal fingerprint needs both Глава AND >=2 Статья."""
        profile = LegalDomainProfile(settings=FakeSettings())
        # Has articles but no chapter → no fingerprint
        text = "Статья 1. Текст.\nСтатья 2. Текст.\n"
        assert profile.structural_fingerprint(text) is False

    def test_classify_score_with_federal_law_bonus(self) -> None:
        """'Федеральный закон' adds +2.0 to legal score."""
        profile = LegalDomainProfile(settings=FakeSettings())
        text = "Федеральный закон о чём-то.\nСтатья 1. Текст.\nСтатья 2. Текст.\nГлава 1. Раздел.\n"
        score = profile.classify_score(text)
        # Base: 0.5*2 + 0.3*1 + 0.2*0 = 1.3, +2.0 for Федеральный закон = 3.3
        assert score >= 3.0

    def test_contract_language_does_not_raise_legal_score(self) -> None:
        """Agreements belong to ContractDomainProfile; only structure counts for legal."""
        profile = LegalDomainProfile(settings=FakeSettings())
        text = "Стороны договорились о следующем.\nСтатья 1. Текст.\nСтатья 2. Текст.\nГлава 1. Раздел.\n"
        assert abs(profile.classify_score(text) - 1.3) < 1e-9


class TestAnswerPolicies:
    def test_versioned_domains_preserve_breadth_and_temporal_rules(self) -> None:
        expected = {
            "legal": (
                "- Обязательно указывай номер статьи/пункта, если он есть в контексте "
                "(например: «Согласно ст. 15 ФЗ-XXX» или «п. 3.2 Договора»).\n"
                "- Не перефразируй формулировки нормативных актов — цитируй максимально близко к тексту.\n"
            ),
            "decree": (
                "- Каждый указ идентифицируй по номеру и дате.\n"
                "- Ссылайся на конкретный пункт постановляющей части, не пересказывай указ целиком.\n"
            ),
        }
        stamp = (
            "- Ответ дан по состоянию на 01.01.2026. "
            "Если подходят разные редакции — укажи, какая редакция использована.\n"
        )

        for domain, rules in expected.items():
            policy = policy_for(domain)
            assert policy.retrieval_fallback_to_full_corpus() is True

            assert policy.prompt_addendum() == rules
            assert policy.prompt_addendum(date(2026, 1, 1)) == rules + stamp

    def test_general_and_unknown_domains_have_no_addendum_or_fallback(self) -> None:
        for domain in ("general", "unknown", ""):
            policy = policy_for(domain)
            assert policy.prompt_addendum() is None
            assert policy.retrieval_fallback_to_full_corpus() is False

    def test_document_profiles_do_not_own_answer_rules(self) -> None:
        for profile in (GeneralDomainProfile(), LegalDomainProfile(), DecreeDomainProfile()):
            assert isinstance(profile, DocumentProfile)
            assert profile.answer_policy is not None
            assert not hasattr(profile, "retrieval_fallback_to_full_corpus")

    def test_rag_addendum_respects_registered_domains(self) -> None:
        ctx = SimpleNamespace(as_of_date=None)
        assert domain_prompt_addendum("legal", cast("RagContext", ctx), None) is None
        assert domain_prompt_addendum("legal", cast("RagContext", ctx), ("general",)) is None
        assert domain_prompt_addendum("unknown", cast("RagContext", ctx), ("unknown",)) is None
        assert domain_prompt_addendum("legal", cast("RagContext", ctx), ("legal",)) == (
            policy_for("legal").prompt_addendum()
        )
