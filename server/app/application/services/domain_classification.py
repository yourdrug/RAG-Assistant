"""Domain classification -- registry-based with legacy-marksman fallback.

Shared by DocumentProcessor (API upload) and document_pipeline.classify_domain
(CLI ingestion): one place decides between DomainProfileRegistry.classify() and
the legacy heuristic, so API and CLI can never drift apart on thresholds.
Ambiguity handling is API-only: the warning is returned, the caller decides
how to surface it (documents.warning_message + metrics + log).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from domain.services.document_domain_classifier import classify_document_domain

if TYPE_CHECKING:
    from application.dto.document_processing_context import ProcessingContext
    from application.ports.document_processing import MetricsCollectorPort
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.domain_profile.settings_port import DomainSettingsPort

log = logging.getLogger("default")


@dataclass(frozen=True)
class DomainClassification:
    """Classification result: domain key + optional ambiguity warning."""

    domain_key: str
    warning: str | None = None
    confidence: float = 1.0
    is_ambiguous: bool = False
    candidate_scores: dict[str, float] = field(default_factory=dict)
    used_registry: bool = False

    @property
    def scores_str(self) -> str:
        return ", ".join(f"{k}={v:.1f}" for k, v in self.candidate_scores.items() if not k.startswith("__"))

    @property
    def ambiguous_candidates(self) -> str:
        return ",".join(sorted(k for k in self.candidate_scores if not k.startswith("__"))) or self.domain_key


def classify_document_text(
    text: str,
    *,
    domain_registry=None,
    domain_settings=None,
    fallback_threshold: float = 2.0,
    legacy_classifier=None,
) -> DomainClassification:
    """Registry-based classification with legacy fallback.

    Returns (domain_key, ambiguous_warning | None). Ambiguous results are
    never swallowed -- the caller surfaces them in documents.warning_message.
    """
    if domain_registry is None or domain_settings is None:
        if legacy_classifier is None:
            log.warning("DomainRegistry unavailable -- no fallback classifier provided")
            return DomainClassification(domain_key="general")
        log.warning("DomainRegistry unavailable -- falling back to legacy classifier")
        return DomainClassification(domain_key=legacy_classifier(text, threshold=fallback_threshold))

    result = domain_registry.classify(text, settings=domain_settings)
    if not (result.is_ambiguous or result.confidence < 0.5):
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


def classify_processing_document(
    ctx: ProcessingContext,
    full_text: str,
    *,
    domain_registry: DomainProfileRegistry | None,
    domain_settings: DomainSettingsPort | None,
    fallback_threshold: float,
    metrics: MetricsCollectorPort,
) -> None:
    """Classify an upload unless its domain was explicitly supplied.

    Record API metrics and warnings without changing the shared CLI classifier.
    """
    if ctx.doc_domain is not None:
        return
    classification = classify_document_text(
        full_text,
        domain_registry=domain_registry,
        domain_settings=domain_settings,
        fallback_threshold=fallback_threshold,
        legacy_classifier=classify_document_domain,
    )
    if classification.used_registry:
        metrics.observe_domain_classification(
            classification.domain_key, "document", classification.confidence
        )
    if classification.warning:
        metrics.inc_domain_ambiguous(classification.ambiguous_candidates)
        log.warning("Ambiguous classification for doc %d: %s", ctx.document_id, classification.scores_str)
        ctx.warnings.append(classification.warning)
    ctx.doc_domain = classification.domain_key
    log.info("Auto-detected doc_domain=%s for doc %d", ctx.doc_domain, ctx.document_id)
