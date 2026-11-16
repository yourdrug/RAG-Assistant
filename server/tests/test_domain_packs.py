"""Domain packs: explicit registry, neutral default, and a core that never imports a pack."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from application.ingestion.profiles import ensure_known_domain, known_domains
from application.runtime.errors import ConfigurationError
from composition.domain_packs import DOMAIN_PACKS, selected_pack
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.shared.exceptions import ValidationError
from domain.shared.profile_policy import AnswerPolicy
from infrastructure.domain_profile import register_pack_profiles
from infrastructure.domain_profile.answer_policy import classify_query_domain, policy_for
from infrastructure.domain_profile.packs import NEUTRAL_PACK, DomainPack
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile, current_evidence_profile
from infrastructure.ml.rag.rag_steps import RagOperations


APP_DIR = Path(__file__).resolve().parents[1] / "app"

# Imports every core module while any domain pack import raises; prints the modules that needed one.
BLOCKED_PACKS_PROBE = """
import importlib, importlib.abc, pkgutil, sys

PACKS = "infrastructure.domain_packs"

class BlockPacks(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == PACKS or name.startswith(PACKS + "."):
            raise ImportError("core imported a domain pack: " + name)

sys.meta_path.insert(0, BlockPacks())

for layer in ("domain", "application", "infrastructure", "presentation"):
    for module in pkgutil.walk_packages(importlib.import_module(layer).__path__, layer + "."):
        if module.name == PACKS or module.name.startswith(PACKS + "."):
            continue

        try:
            importlib.import_module(module.name)
        except Exception as exc:  # Optional backends (surya, alembic env) fail for unrelated reasons.
            if "domain pack" in str(exc):
                print(module.name, exc)
"""


def registry_for(pack: DomainPack) -> DomainProfileRegistry:
    registry = DomainProfileRegistry()
    register_pack_profiles(registry, pack, DomainSettingsAdapter())
    return registry


def test_core_modules_import_without_any_domain_pack() -> None:
    result = subprocess.run(  # noqa: S603 - fixed interpreter and inline probe, no user input.
        [sys.executable, "-c", BLOCKED_PACKS_PROBE],
        cwd=APP_DIR,
        env={**os.environ, "PYTHONPATH": str(APP_DIR)},
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    assert result.stdout == ""


def test_unknown_domain_pack_is_a_configuration_error() -> None:
    assert selected_pack(NEUTRAL_PACK.key) is NEUTRAL_PACK

    with pytest.raises(ConfigurationError, match="belarus_typo"):
        selected_pack("belarus_typo")


def test_neutral_pack_knows_only_general_documents_and_rules() -> None:
    registry = registry_for(NEUTRAL_PACK)
    assert known_domains(registry) == ("general",)
    assert registry.doc_type_keywords == ()
    assert registry.operative_part_profile() is None
    assert isinstance(NEUTRAL_PACK.evidence_profile, GenericEvidenceProfile)
    assert classify_query_domain("Статья 15 закона", registry) == "general"
    assert policy_for("legal", registry) == AnswerPolicy()

    with pytest.raises(ValidationError, match="legal"):
        ensure_known_domain(registry, "legal")


@pytest.mark.belarus
def test_law_pack_registers_its_profiles_keywords_and_evidence_rules() -> None:
    pack = DOMAIN_PACKS["belarus_law"]
    registry = registry_for(pack)
    assert known_domains(registry) == ("general", "legal", "decree", "contract")
    assert dict(registry.doc_type_keywords)["contract"] == ("договор", "контракт", "соглашение")
    assert registry.operative_part_profile() is registry.get("decree")
    assert pack.evidence_profile.key == "belarus"
    assert all(path.exists() for path in pack.benchmark_sets)
    ensure_known_domain(registry, "decree")


@pytest.mark.asyncio
async def test_new_pack_needs_only_its_definition_and_one_registration() -> None:
    class FaqProfile(GeneralDomainProfile):
        key = "faq"
        query_patterns: tuple[str, ...] = (r"как\s+сбросить",)
        query_priority = 1
        answer_policy = AnswerPolicy("- Ссылайся на статью базы знаний.\n")

    class FaqEvidenceProfile(GenericEvidenceProfile):
        key = "faq"

    pack = DomainPack(
        key="support",
        language="ru",
        document_profiles=(lambda _settings: GeneralDomainProfile(), lambda _settings: FaqProfile()),
        evidence_profile=FaqEvidenceProfile(),
        doc_type_keywords=(("faq", ("вопрос",)),),
    )
    registry = registry_for(pack)
    assert known_domains(registry) == ("general", "faq")
    assert classify_query_domain("Как сбросить пароль?", registry) == "faq"
    assert policy_for("faq", registry).rules == "- Ссылайся на статью базы знаний.\n"
    operations = RagOperations(
        MagicMock(), document_access=MagicMock(), evidence_profile=pack.evidence_profile
    )

    async with operations.corpus():
        assert current_evidence_profile() is pack.evidence_profile
