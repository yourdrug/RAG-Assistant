"""Internal knowledge pack: employee instructions and procedures."""

from pathlib import Path

from infrastructure.domain_packs.internal.domain.profiles import (
    InternalInstructionProfile,
    InternalProcedureProfile,
)
from infrastructure.domain_packs.internal.evidence.profile import InternalEvidenceProfile
from infrastructure.domain_profile.packs import DomainPack, general_profile


BENCHMARKS_DIR = Path(__file__).resolve().parent / "benchmarks"


INTERNAL_PACK = DomainPack(
    key="internal",
    language="ru",
    document_profiles=(general_profile, InternalInstructionProfile, InternalProcedureProfile),
    evidence_profile=InternalEvidenceProfile(),
    doc_type_keywords=(
        ("internal_procedure", ("регламент", "положение", "порядок", "политика")),
        ("internal_instruction", ("инструкция", "руководство", "памятка")),
    ),
    benchmark_sets=(BENCHMARKS_DIR / "chunking",),
)
