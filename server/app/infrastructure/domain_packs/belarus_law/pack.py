"""Belarusian law domain pack: one registration in Composition enables it."""

from __future__ import annotations

from pathlib import Path

from domain.ingestion.profiles.contract import ContractDomainProfile
from infrastructure.domain_packs.belarus_law.domain.decree import DecreeDomainProfile
from infrastructure.domain_packs.belarus_law.domain.doc_types import DOC_TYPE_KEYWORDS
from infrastructure.domain_packs.belarus_law.domain.legal import LegalDomainProfile
from infrastructure.domain_packs.belarus_law.evidence.profile import BelarusEvidenceProfile
from infrastructure.domain_packs.belarus_law.persistence.repositories import sql_act_repositories
from infrastructure.domain_profile.packs import DomainPack, general_profile


BENCHMARKS_DIR = Path(__file__).resolve().parent / "benchmarks"

BELARUS_LAW_PACK = DomainPack(
    key="belarus_law",
    language="ru",
    document_profiles=(general_profile, LegalDomainProfile, DecreeDomainProfile, ContractDomainProfile),
    evidence_profile=BelarusEvidenceProfile(),
    doc_type_keywords=DOC_TYPE_KEYWORDS,
    benchmark_sets=(BENCHMARKS_DIR / "benchmark-questions-v2.json", BENCHMARKS_DIR / "chunking"),
    repositories=sql_act_repositories,
)
