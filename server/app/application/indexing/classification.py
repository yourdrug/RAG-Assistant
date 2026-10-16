"""Shared registry-based domain classification with a legacy fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from application.indexing.dto import DomainClassification


if TYPE_CHECKING:
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.domain_profile.settings_port import DomainSettingsPort

log = logging.getLogger("default")
MIN_CLASSIFICATION_CONFIDENCE = 0.5


class LegacyClassifier(Protocol):
    def __call__(self, text: str, *, threshold: float) -> str: ...


def classify_document_text(
    text: str,
    *,
    domain_registry: DomainProfileRegistry | None = None,
    domain_settings: DomainSettingsPort | None = None,
    fallback_threshold: float = 2.0,
    legacy_classifier: LegacyClassifier | None = None,
) -> DomainClassification:
    if domain_registry is None or domain_settings is None:
        if legacy_classifier is None:
            log.warning("DomainRegistry unavailable -- no fallback classifier provided")
            return DomainClassification(domain_key="general")

        log.warning("DomainRegistry unavailable -- falling back to legacy classifier")
        return DomainClassification(domain_key=legacy_classifier(text, threshold=fallback_threshold))

    result = domain_registry.classify(text, settings=domain_settings)

    if not (result.is_ambiguous or result.confidence < MIN_CLASSIFICATION_CONFIDENCE):
        return DomainClassification(
            domain_key=result.domain_key,
            confidence=result.confidence,
            is_ambiguous=result.is_ambiguous,
            candidate_scores=dict(result.candidate_scores),
            used_registry=True,
        )

    scores_str = ", ".join(
        f"{k}={v:.1f}" for k, v in result.candidate_scores.items() if not k.startswith("__")
    )
    return DomainClassification(
        domain_key=result.domain_key,
        warning=f"Классификация домена неоднозначна ({scores_str}) — требуется ручная проверка",
        confidence=result.confidence,
        is_ambiguous=result.is_ambiguous,
        candidate_scores=dict(result.candidate_scores),
        used_registry=True,
    )
