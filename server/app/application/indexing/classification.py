"""Shared registry-based domain classification with an explicit general fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.indexing.dto import DomainClassification
from domain.knowledge.enums import DocDomain


if TYPE_CHECKING:
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.registry import DomainProfileRegistry

log = logging.getLogger("default")
MIN_CLASSIFICATION_CONFIDENCE = 0.5


def classify_document_text(
    text: str,
    *,
    domain_registry: DomainProfileRegistry | None = None,
    domain_settings: DomainSettingsPort | None = None,
) -> DomainClassification:
    if domain_registry is None or domain_settings is None:
        log.warning("Domain registry or settings unavailable -- using general domain")
        return DomainClassification(domain_key=DocDomain.GENERAL.value)

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
