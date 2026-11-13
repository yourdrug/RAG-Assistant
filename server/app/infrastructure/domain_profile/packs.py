"""Domain pack contract and the neutral pack built into the core.

A domain pack bundles the rules of one subject area: document profiles (with their
answer policies), evidence rules, document-type keywords, language and benchmark sets.
Packs live under ``packs/<name>/`` and are registered explicitly by Composition; the
core never imports a pack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.ingestion.profiles.general import GeneralDomainProfile
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from domain.ingestion.doc_type import DocTypeKeywords
    from domain.ingestion.ports import DomainSettingsPort
    from domain.ingestion.profiles.protocol import DocumentProfile
    from infrastructure.ml.evidence_profiles.profiles import EvidenceProfile


type DocumentProfileFactory = Callable[[DomainSettingsPort | None], DocumentProfile]


NEUTRAL_PACK_KEY = "generic"


@dataclass(frozen=True)
class DomainPack:
    key: str
    language: str
    # Every pack includes ``general``: it is the fallback of document classification.
    document_profiles: tuple[DocumentProfileFactory, ...]
    evidence_profile: EvidenceProfile
    doc_type_keywords: DocTypeKeywords = ()
    benchmark_sets: tuple[Path, ...] = ()

    def build_profiles(self, settings: DomainSettingsPort | None) -> list[DocumentProfile]:
        return [factory(settings) for factory in self.document_profiles]


def general_profile(_settings: DomainSettingsPort | None) -> DocumentProfile:
    return GeneralDomainProfile()


NEUTRAL_PACK = DomainPack(
    key=NEUTRAL_PACK_KEY,
    language="ru",
    document_profiles=(general_profile,),
    evidence_profile=GenericEvidenceProfile(),
)
