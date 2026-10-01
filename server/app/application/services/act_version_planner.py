"""Prepare extracted version metadata and apply the configured date trust rule."""

from __future__ import annotations

import logging
from datetime import date

from domain.domain_profile.protocol import DomainProfile, refs_to_metadata
from domain.domain_profile.settings_port import DomainSettingsPort

from application.dto.versioning_dto import VersioningPlan

log = logging.getLogger("default")
_DATE_SOURCE_EXTRACTED = "extracted"
_DATE_SOURCE_EXTRACTED_TRUSTED = "extracted_trusted"


class ActVersionPlanner:
    def __init__(self, settings: DomainSettingsPort) -> None:
        self._settings = settings

    def resolve_date_source(
        self,
        domain_key: str,
        effective_date: date | None,
        date_confidence: float | None,
    ) -> str:
        if effective_date is None:
            return _DATE_SOURCE_EXTRACTED
        # Seeding guarantees the key exists — a missing key is a loud config error.
        threshold = float(self._settings.get("effective_date_auto_trust_threshold", domain_key=domain_key))
        if date_confidence is not None and date_confidence >= threshold:
            return _DATE_SOURCE_EXTRACTED_TRUSTED
        return _DATE_SOURCE_EXTRACTED

    async def prepare_document_versioning(
        self, profile: DomainProfile, full_text: str
    ) -> VersioningPlan | None:
        """Extract version metadata without committing a version row."""
        if not profile.is_versioned:
            return None
        try:
            refs = profile.extract_references(full_text) or []
            date_candidate = profile.extract_effective_date(full_text)
            raw_effective_date = (
                date_candidate.effective_from if date_candidate and date_candidate.effective_from else None
            )
            effective_from = None
            if raw_effective_date is not None and date_candidate is not None:
                threshold = float(
                    self._settings.get("effective_date_auto_trust_threshold", domain_key=profile.key)
                )
                if date_candidate.confidence >= threshold:
                    effective_from = raw_effective_date
            return VersioningPlan(
                extracted_refs=refs,
                domain_metadata=refs_to_metadata(refs) if refs else None,
                effective_date=raw_effective_date,
                effective_from=effective_from,
                date_confidence=date_candidate.confidence if date_candidate else None,
            )
        except Exception as e:
            log.warning("Versioning extraction failed for domain %s: %s", profile.key, e)
            return VersioningPlan([], None, None, None, None, f"Версионирование не выполнено: {e}")
