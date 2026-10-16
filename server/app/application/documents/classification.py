"""Domain classification -- registry-based with legacy-marksman fallback.

Shared by DocumentProcessor (API upload) and document_pipeline.classify_domain
(CLI ingestion): one place decides between DomainProfileRegistry.classify() and
the legacy heuristic, so API and CLI can never drift apart on thresholds.
Ambiguity handling is API-only: the warning is returned, the caller decides
how to surface it (documents.warning_message + metrics + log).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from application.indexing import DomainClassification, classify_document_text
from domain.services.document_domain_classifier import classify_document_domain


if TYPE_CHECKING:
    from application.documents.processing_context import ProcessingContext
    from application.documents.processing_ports import MetricsCollectorPort
    from domain.domain_profile.registry import DomainProfileRegistry
    from domain.domain_profile.settings_port import DomainSettingsPort

log = logging.getLogger("default")


def classify_processing_document(
    ctx: ProcessingContext,
    full_text: str,
    *,
    domain_registry: DomainProfileRegistry | None,
    domain_settings: DomainSettingsPort | None,
    fallback_threshold: float,
    metrics: MetricsCollectorPort,
) -> None:
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


__all__ = ["DomainClassification", "classify_document_text", "classify_processing_document"]
