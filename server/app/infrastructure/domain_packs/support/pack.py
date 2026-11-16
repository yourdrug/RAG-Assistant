"""Support pack: customer FAQs and product instructions."""

from pathlib import Path

from infrastructure.domain_packs.support.domain.profiles import SupportFaqProfile, SupportGuideProfile
from infrastructure.domain_packs.support.evidence.profile import SupportEvidenceProfile
from infrastructure.domain_profile.packs import DomainPack, general_profile


BENCHMARKS_DIR = Path(__file__).resolve().parent / "benchmarks"


SUPPORT_PACK = DomainPack(
    key="support",
    language="ru",
    document_profiles=(general_profile, SupportFaqProfile, SupportGuideProfile),
    evidence_profile=SupportEvidenceProfile(),
    doc_type_keywords=(
        ("support_faq", ("faq", "часто задаваемые вопросы", "вопросы и ответы")),
        ("support_guide", ("инструкция", "руководство", "настройка", "создание")),
    ),
    benchmark_sets=(BENCHMARKS_DIR / "questions.json", BENCHMARKS_DIR / "chunking"),
)
