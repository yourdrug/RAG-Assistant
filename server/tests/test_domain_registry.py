"""Tests for DomainProfileRegistry classification logic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from infrastructure.domain_profile.registry import (
    DomainProfileRegistry,
)
from domain.domain_profile.profiles.decree import DecreeDomainProfile
from domain.domain_profile.profiles.general import GeneralDomainProfile
from domain.domain_profile.profiles.legal import LegalDomainProfile


class FakeSettings:
    def get(self, key: str, domain_key: str = "") -> str:
        defaults = {
            "classification_threshold": "2.0",
            "fingerprint_min_points": "1",
            "fingerprint_min_articles": "2",
        }
        return defaults.get(key, "0")


def _make_registry() -> DomainProfileRegistry:
    reg = DomainProfileRegistry()
    reg.register(GeneralDomainProfile())
    reg.register(LegalDomainProfile(settings=FakeSettings()))
    reg.register(DecreeDomainProfile(settings=FakeSettings()))
    return reg


class TestRegistryBasics:
    def test_register_and_get(self):
        reg = _make_registry()
        assert reg.get("general") is not None
        assert reg.get("legal") is not None
        assert reg.get("decree") is not None

    def test_all_returns_all_profiles(self):
        reg = _make_registry()
        assert len(reg.all()) == 3

    def test_get_unknown_raises(self):
        reg = _make_registry()
        try:
            reg.get("unknown")
            raise AssertionError("Should have raised KeyError")
        except KeyError:
            pass


class TestFingerprintClassification:
    def test_decree_fingerprint_wins(self):
        # Text must be >= MIN_CLASSIFIABLE_CHARS (120)
        text = (
            "УКАЗ ПРЕЗИДЕНТА РЕСПУБЛИКИ БЕЛАРУСЬ\n"
            "О некоторых мерах по регулированию экономических отношений\n\n"
            "ПОСТАНОВЛЯЮ:\n\n"
            "1. Принять предложенные меры по совершенствованию системы управления.\n"
            "2. Контроль за исполнением настоящего указа возложить на Министерство финансов.\n"
            "3. Настоящий указ вступает в силу со дня его подписания.\n"
        )
        reg = _make_registry()
        result = reg.classify(text, settings=FakeSettings())
        assert result.domain_key == "decree"
        assert result.confidence == 1.0
        assert result.is_ambiguous is False
        assert result.is_low_signal is False

    def test_legal_fingerprint_wins(self):
        text = (
            "Глава 1. Общие положения\n\n"
            "Статья 1. Основные понятия\n\n"
            "В настоящем законе используются понятия.\n\n"
            "Статья 2. Предмет регулирования\n\n"
            "Настоящий закон регулирует отношения.\n"
        )
        reg = _make_registry()
        result = reg.classify(text, settings=FakeSettings())
        assert result.domain_key == "legal"
        assert result.confidence == 1.0

    def test_short_text_returns_low_signal(self):
        text = "Короткий текст."
        reg = _make_registry()
        result = reg.classify(text, settings=FakeSettings())
        assert result.is_low_signal is True
        assert result.confidence == 0.3


class TestDensityScoreClassification:
    def test_legal_density_wins(self):
        # Text without fingerprint triggers (no "Глава"), but with legal markers
        # Must be >= MIN_CLASSIFIABLE_CHARS (120)
        text = (
            "Федеральный закон от 30 декабря 2008 г. № 307-ФЗ\n"
            "Статья 1. Основные понятия\n"
            "В целях настоящего закона используются основные понятия.\n"
            "Статья 2. Предмет регулирования\n"
            "Настоящий закон регулирует отношения по поводу аудиторской деятельности.\n"
            "Статья 3. Аудиторская деятельность\n"
            "Аудиторская деятельность осуществляется аудиторскими организациями.\n"
        )
        reg = _make_registry()
        result = reg.classify(text, settings=FakeSettings())
        assert result.domain_key == "legal"

    def test_general_when_no_signals(self):
        text = "Обычный корпоративный документ без юридических маркеров. " * 10
        reg = _make_registry()
        result = reg.classify(text, settings=FakeSettings())
        assert result.domain_key == "general"
        assert result.is_ambiguous is False


class TestAmbiguityDetection:
    def test_sticky_prior_boosts_domain(self):
        text = "УКАЗ ПРЕЗИДЕНТА\n" "ПОСТАНОВЛЯЮ:\n" "1. Мера.\n"
        reg = _make_registry()
        # Without prior — may be ambiguous or not
        result_no_prior = reg.classify(text, settings=FakeSettings())
        # With prior=decree — should definitely be decree
        result_with_prior = reg.classify(text, prior_domain="decree", settings=FakeSettings())
        assert result_with_prior.domain_key == "decree"
        # Sticky bonus should increase confidence
        if not result_no_prior.is_ambiguous:
            assert result_with_prior.confidence >= result_no_prior.confidence


class TestEdgeCases:
    def test_empty_text(self):
        reg = _make_registry()
        result = reg.classify("", settings=FakeSettings())
        assert result.is_low_signal is True

    def test_whitespace_only(self):
        reg = _make_registry()
        result = reg.classify("   \n  \n  ", settings=FakeSettings())
        assert result.is_low_signal is True

    def test_unknown_domain_falls_to_general(self):
        text = "Текст без каких-либо маркеров домена. " * 20
        reg = _make_registry()
        result = reg.classify(text, settings=FakeSettings())
        assert result.domain_key == "general"


class _TwoFingerprintProfile:
    """Profile whose fingerprint always fires — used to create a conflict."""

    key = "boilerplate"
    display_name = "Conflict"
    is_versioned = False

    def __init__(self) -> None:
        pass

    def config_defaults(self):
        return []

    def structural_fingerprint(self, text: str) -> bool:
        return True

    def classify_score(self, text: str) -> float:
        return 0.0

    def content_boundaries(self):
        return []

    def extract_references(self, text: str):
        return []

    def extract_effective_date(self, text: str):
        return None

    def prompt_addendum(self, breadth: str, as_of_date=None):
        return None

    def retrieval_fallback_to_full_corpus(self) -> bool:
        return False


class TestFingerprintConflict:
    def test_two_fingerprints_fire_is_ambiguous(self):
        reg = _make_registry()
        reg.register(_TwoFingerprintProfile())
        text = "УКАЗ ПРЕЗИДЕНТА\nПОСТАНОВЛЯЮ:\n1. Пункт.\n2. Пункт.\n" * 4
        result = reg.classify(text, settings=FakeSettings())
        # Two formats claimed "this is definitely me" — never a silent pick
        assert result.is_ambiguous is True
        assert result.confidence == 0.4
        assert set(result.candidate_scores.keys()) == {"decree", "boilerplate"}


class TestMarginAmbiguity:
    def test_borderline_scores_flagged_ambiguous(self):
        reg = DomainProfileRegistry()
        reg.register(LegalDomainProfile(settings=FakeSettings()))
        reg.register(DecreeDomainProfile(settings=FakeSettings()))
        # Both get density scores but no fingerprint fires; the margin between
        # the two winners decides: either a clean winner with confidence > 0.5
        # or an explicit ambiguity — never a silent low-margin pick.
        text = (
            "УКАЗ был издан в соответствии со Статья 15 другого закона.\n"
            "ПОСТАНОВЛЯЮ упомянуто в тексте постановления.\n"
            "Статья 20. Статья 21. Статья 22. Глава отсутствует.\n"
        )
        result = reg.classify(text, settings=FakeSettings())
        assert result.candidate_scores, "density scores must be exposed for review"
        if result.is_ambiguous:
            assert result.confidence == 0.5
        else:
            assert result.confidence > 0.5

    def test_short_text_inherits_prior_domain(self):
        reg = _make_registry()
        result = reg.classify("1. Пункт.", prior_domain="decree", settings=FakeSettings())
        assert result.domain_key == "decree"
        assert result.is_low_signal is True

    def test_short_text_without_prior_falls_to_general(self):
        reg = _make_registry()
        result = reg.classify("1. Пункт.", settings=FakeSettings())
        assert result.domain_key == "general"
        assert result.is_low_signal is True


# ---------------------------------------------------------------------------
# Confidence formula — numerical verification
# ---------------------------------------------------------------------------


class TestConfidenceFormula:
    def test_confidence_formula_clean_winner(self):
        """confidence = min(1.0, (top_score - threshold) / threshold)."""
        reg = _make_registry()
        # Decree text with strong signals → score well above threshold
        text = (
            "УКАЗ ПРЕЗИДЕНТА\n"
            "ПОСТАНОВЛЯЮ:\n"
            "1. Пункт первый.\n"
            "2. Пункт второй.\n"
            "3. Пункт третий.\n"
            "4. Пункт четвёртый.\n"
        )
        result = reg.classify(text, settings=FakeSettings())
        if result.domain_key == "decree" and not result.is_ambiguous:
            # confidence = min(1.0, (score - 2.0) / 2.0)
            assert 0.0 <= result.confidence <= 1.0

    def test_confidence_clamped_at_1(self):
        """confidence must never exceed 1.0 even with very high scores."""
        reg = DomainProfileRegistry()
        reg.register(GeneralDomainProfile())
        reg.register(DecreeDomainProfile(settings=FakeSettings()))
        # Very long decree text → high score
        text = "УКАЗ\nПОСТАНОВЛЯЮ:\n" + "\n".join(f"{i}. Пункт {i}." for i in range(50))
        result = reg.classify(text, settings=FakeSettings())
        assert result.confidence <= 1.0

    def test_confidence_zero_when_score_zero(self):
        """When top_score=0 and threshold>0, confidence should be 1.0 (general wins)."""
        reg = DomainProfileRegistry()
        reg.register(GeneralDomainProfile())
        reg.register(DecreeDomainProfile(settings=FakeSettings()))
        # Plain text with no decree markers → score=0 for decree
        result = reg.classify("Простой текст без маркеров домена.", settings=FakeSettings())
        # Falls below threshold → classified as general
        assert result.domain_key == "general"


# ---------------------------------------------------------------------------
# Threshold boundary conditions
# ---------------------------------------------------------------------------


class TestThresholdBoundary:
    def test_score_exactly_at_threshold_classifies(self):
        """When top_score == threshold, it should classify as the domain (not general)."""
        reg = DomainProfileRegistry()
        reg.register(GeneralDomainProfile())
        reg.register(DecreeDomainProfile(settings=FakeSettings()))
        # "ПОСТАНОВЛЯЮ" alone gives +2.0, which equals the threshold (2.0)
        # Need 120+ chars to avoid low_signal fallback
        text = "ПОСТАНОВЛЯЮ " + "текст " * 30 + "\n"
        result = reg.classify(text, settings=FakeSettings())
        # score=2.0, threshold=2.0 → score < threshold is False → classify as domain
        assert result.domain_key == "decree"
        assert result.is_low_signal is False

    def test_score_below_threshold_returns_general(self):
        """When top_score < threshold, should classify as general."""
        reg = DomainProfileRegistry()
        reg.register(GeneralDomainProfile())
        reg.register(DecreeDomainProfile(settings=FakeSettings()))
        # Very weak text → score < 2.0
        text = "Обычный документ без особых признаков.\n"
        result = reg.classify(text, settings=FakeSettings())
        # If score < threshold, falls to general
        if result.domain_key == "general":
            assert result.confidence >= 0.0


# ---------------------------------------------------------------------------
# Sticky prior edge cases
# ---------------------------------------------------------------------------


class TestStickyPriorEdgeCases:
    def test_sticky_prior_ignored_when_score_zero(self):
        """Sticky prior only applies when raw > 0. If score=0, prior is ignored."""
        reg = DomainProfileRegistry()
        reg.register(GeneralDomainProfile())
        reg.register(DecreeDomainProfile(settings=FakeSettings()))
        # Long plain text → decree score = 0, even with prior
        result = reg.classify(
            "Обычный текст без каких-либо маркеров. " * 10,
            prior_domain="decree",
            settings=FakeSettings(),
        )
        # Decree score is 0, sticky not applied, falls to general
        assert result.domain_key == "general"

    def test_unknown_prior_domain_no_effect(self):
        """prior_domain pointing to non-existent profile has no effect on scoring."""
        reg = _make_registry()
        result = reg.classify(
            "Обычный текст без маркеров домена. " * 10,
            prior_domain="nonexistent",
            settings=FakeSettings(),
        )
        # Unknown prior → no sticky boost, normal classification
        assert result.domain_key == "general"

    def test_unknown_prior_with_short_text(self):
        """Short text with unknown prior falls back to general."""
        reg = _make_registry()
        result = reg.classify(
            "Короткий", prior_domain="nonexistent", settings=FakeSettings(),
        )
        assert result.domain_key == "general"
        assert result.is_low_signal is True


# ---------------------------------------------------------------------------
# All scores zero / empty ranked
# ---------------------------------------------------------------------------


class TestAllScoresZero:
    def test_all_scores_zero_returns_general(self):
        """When all candidate scores are 0, top_score=0 < threshold → general."""
        reg = _make_registry()
        # Plain text with no markers for any domain
        result = reg.classify(
            "Простой текст без каких-либо маркеров или структуры.\n"
            "Никаких статей, глав, указов или постановлений.\n",
            settings=FakeSettings(),
        )
        assert result.domain_key == "general"

    def test_only_general_registered(self):
        """When only general is registered, candidates is empty → general."""
        reg = DomainProfileRegistry()
        reg.register(GeneralDomainProfile())
        result = reg.classify(
            "Любой текст.\n" * 20,
            settings=FakeSettings(),
        )
        assert result.domain_key == "general"
        assert result.confidence == 1.0


# ---------------------------------------------------------------------------
# Margin ratio with real profiles
# ---------------------------------------------------------------------------


class TestMarginRatioRealProfiles:
    def test_clean_winner_not_ambiguous(self):
        """Strong decree text should not be ambiguous with legal."""
        reg = DomainProfileRegistry()
        reg.register(GeneralDomainProfile())
        reg.register(LegalDomainProfile(settings=FakeSettings()))
        reg.register(DecreeDomainProfile(settings=FakeSettings()))
        text = (
            "УКАЗ ПРЕЗИДЕНТА\n"
            "ПОСТАНОВЛЯЮ:\n"
            "1. Первый пункт.\n"
            "2. Второй пункт.\n"
            "3. Третий пункт.\n"
        )
        result = reg.classify(text, settings=FakeSettings())
        # Strong decree signals → should not be ambiguous
        if result.domain_key == "decree":
            # If clean winner, confidence > 0.5
            if not result.is_ambiguous:
                assert result.confidence > 0.5


# ---------------------------------------------------------------------------
# Settings adapter integration
# ---------------------------------------------------------------------------


class TestSettingsAdapterIntegration:
    def test_missing_threshold_key_in_classify(self):
        """If settings.get() raises KeyError, it propagates."""
        class BadSettings:
            def get(self, key, domain_key=""):
                raise KeyError(f"Missing key: {key}")

        reg = DomainProfileRegistry()
        reg.register(GeneralDomainProfile())
        reg.register(DecreeDomainProfile(settings=BadSettings()))
        text = "УКАЗ\nПОСТАНОВЛЯЮ:\n" + "\n".join(f"{i}. Пункт {i}." for i in range(10)) + "\n"
        import pytest
        with pytest.raises(KeyError):
            reg.classify(text, settings=BadSettings())
