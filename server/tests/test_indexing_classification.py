"""API and CLI use the same registry classification and explicit fallback."""

# ruff: noqa: RUF001 — Russian document fixtures.

import logging

import pytest

from application.indexing.classification import classify_document_text
from application.indexing.pipeline import classify_domain
from domain.ingestion.doc_type import classify_doc_type
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry


class ClassificationSettings:
    def get(self, key: str, domain_key: str) -> str:
        assert domain_key == "legal"
        return {"classification_threshold": "2.0", "fingerprint_min_articles": "2"}[key]


@pytest.mark.parametrize("missing", ["registry", "settings", "both"])
def test_missing_classification_dependencies_use_general(
    missing: str, caplog: pytest.LogCaptureFixture
) -> None:
    registry = None if missing in {"registry", "both"} else DomainProfileRegistry()
    settings = None if missing in {"settings", "both"} else ClassificationSettings()
    with caplog.at_level(logging.WARNING):
        result = classify_document_text(
            "Федеральный закон", domain_registry=registry, domain_settings=settings
        )
    assert result.domain_key == "general"
    assert not result.used_registry
    assert "using general domain" in caplog.text
    assert (
        classify_domain("Федеральный закон", domain_registry=registry, domain_settings=settings)
        == result.domain_key
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Федеральный закон", "general"),
        (
            (
                "Глава 1. Общие положения\n\nСтатья 1. Основные понятия\n\n"
                "В настоящем законе используются понятия.\n\nСтатья 2. Предмет регулирования\n\n"
                "Настоящий закон регулирует отношения.\n"
            ),
            "legal",
        ),
    ],
)
def test_api_and_cli_classify_through_registry(text: str, expected: str) -> None:
    settings = ClassificationSettings()
    registry = DomainProfileRegistry()
    registry.register(GeneralDomainProfile())
    registry.register(LegalDomainProfile(settings=settings))
    result = classify_document_text(text, domain_registry=registry, domain_settings=settings)
    assert result.domain_key == expected
    assert result.used_registry
    assert classify_domain(text, domain_registry=registry, domain_settings=settings) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("ПРИКАЗ о приёме", "order"), ("Договор поставки", "contract"), ("", None)],
)
def test_document_type_classification_is_preserved(text: str, expected: str | None) -> None:
    assert classify_doc_type(text) == expected
