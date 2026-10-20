"""Registry-based domain classification shared by API and CLI ingestion.

DocumentProcessor (API upload) and indexing.classify_domain (CLI ingestion)
use the same classification decision and fallback.
Ambiguity handling is API-only: the warning is returned, the caller decides
how to surface it (documents.warning_message + metrics + log).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.indexing import DomainClassification, classify_document_text


if TYPE_CHECKING:
    from application.documents.ports.processing import MetricsCollectorPort
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.domain_profile.settings_port import DomainSettingsPort

log = logging.getLogger("default")


def classify_processing_document(
    document_id: int,
    full_text: str,
    doc_domain: str | None,
    *,
    domain_registry: DomainProfileRegistry | None,
    domain_settings: DomainSettingsPort | None,
    metrics: MetricsCollectorPort,
) -> DomainClassification:
    if doc_domain is not None:
        return DomainClassification(domain_key=doc_domain)

    classification = classify_document_text(
        full_text,
        domain_registry=domain_registry,
        domain_settings=domain_settings,
    )

    if classification.used_registry:
        metrics.observe_domain_classification(
            classification.domain_key, "document", classification.confidence
        )

    if classification.warning:
        metrics.inc_domain_ambiguous(classification.ambiguous_candidates)
        log.warning("Ambiguous classification for doc %d: %s", document_id, classification.scores_str)

    log.info("Auto-detected doc_domain=%s for doc %d", classification.domain_key, document_id)
    return classification


__all__ = ["classify_processing_document"]
