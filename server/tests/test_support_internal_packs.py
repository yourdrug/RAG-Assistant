"""Pack contracts using illustrative texts, not acceptance benchmark datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from composition.domain_packs import PACK_REGISTRATIONS, loaded_packs, selected_pack
from domain.ingestion.raw_document import RawDocument
from infrastructure.domain_profile import pack_registry, union_registry
from infrastructure.domain_profile.answer_policy import classify_query_domain, policy_for
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.evidence_profiles.profiles import current_evidence_profile, evidence_profile_scope
from infrastructure.ml.langchain_document_parser import LangchainDocumentSplitter


if TYPE_CHECKING:
    from domain.ingestion.profiles.registry import DomainProfileRegistry


pytestmark = pytest.mark.skipif(
    "support" not in PACK_REGISTRATIONS or "internal" not in PACK_REGISTRATIONS,
    reason="Optional packs are disabled in the core-only run",
)


def configured_registry(pack_key: str) -> tuple[DomainProfileRegistry, DomainSettingsAdapter]:
    pack = selected_pack(pack_key)
    settings = DomainSettingsAdapter()
    settings.replace_snapshot(
        {
            (item.key, profile.key): item.value
            for profile in pack.build_profiles(settings)
            for item in profile.config_defaults()
        }
    )
    return pack_registry(pack, settings), settings


@pytest.mark.parametrize(
    "pack_key,keys",
    [
        ("support", ("general", "support_faq", "support_guide")),
        ("internal", ("general", "internal_instruction", "internal_procedure")),
    ],
)
def test_packs_load_without_extra_wiring_and_only_classify_their_profiles(
    pack_key: str, keys: tuple[str, ...]
) -> None:
    loaded = loaded_packs()[pack_key]
    registry, settings = configured_registry(pack_key)
    assert tuple(profile.key for profile in registry.all()) == keys
    assert loaded.wiring.document_hooks is None
    assert loaded.wiring.routers() == ()
    assert registry.operative_part_profile() is None
    assert all(not profile.is_versioned for profile in registry.all())
    assert all(profile.resolution_strategy == "replace" for profile in registry.all())
    unknown = "Отчёт об использовании ресурсов за отчётный период. " * 10
    assert registry.classify(unknown, settings=settings).domain_key == "general"
    assert loaded.pack.evidence_profile.chunk_selection(None) is None


@pytest.mark.parametrize(
    "pack_key,title,expected",
    [
        ("support", "FAQ продукта", "support_faq"),
        ("support", "Инструкция по подключению", "support_guide"),
        ("internal", "Памятка сотрудника", "internal_instruction"),
        ("internal", "Регламент согласования заявок", "internal_procedure"),
    ],
)
def test_explicit_titles_classify_without_triggering_on_mentions(
    pack_key: str, title: str, expected: str
) -> None:
    registry, settings = configured_registry(pack_key)
    body = "Описание области применения и условий выполнения действий. " * 5
    assert registry.classify(f"# {title}\n{body}", settings=settings).domain_key == expected
    assert registry.classify(f"Отчёт\n{body}\n{title}", settings=settings).domain_key == "general"


@pytest.mark.parametrize(
    "pack_key,question,expected",
    [
        ("support", "Как настроить подключение?", "support_guide"),
        ("support", "Что означает ошибка?", "support_faq"),
        ("internal", "Как настроить рабочее место?", "internal_instruction"),
        ("internal", "Кто согласовывает заявку и в какой срок?", "internal_procedure"),
    ],
)
def test_answer_rules_follow_the_space_pack(pack_key: str, question: str, expected: str) -> None:
    registry, _settings = configured_registry(pack_key)
    assert classify_query_domain(question, registry) == expected
    policy = policy_for(expected, registry)
    assert policy.rules
    assert not policy.retrieval_fallback_to_full_corpus()
    assert policy.as_of_note is None
    assert policy_for("legal", registry).rules is None


def test_support_faq_preserves_question_and_answer_pairs() -> None:
    registry, settings = configured_registry("support")
    text = (
        "FAQ продукта\nВопрос: Как подключиться?\nОтвет: Откройте настройки подключения.\n"
        "Вопрос: Почему нет связи?\nОтвет: Проверьте состояние сети.\n"
    )
    chunks = LangchainDocumentSplitter(registry, settings).split(
        [RawDocument(text, {"source": "faq.txt"})], domain="support_faq"
    )
    bodies = [chunk.page_content for chunk in chunks]
    assert any("Как подключиться?" in body and "Откройте настройки" in body for body in bodies)
    assert any("Почему нет связи?" in body and "Проверьте состояние" in body for body in bodies)
    assert not any("Как подключиться?" in body and "Почему нет связи?" in body for body in bodies)


def test_profile_keys_do_not_overwrite_other_packs_in_admin_registry() -> None:
    registries = [configured_registry(key)[0] for key in ("support", "internal")]
    union = union_registry(registries)
    assert len(union.all()) == 5
    assert union.get("support_guide").display_name == "Инструкции поддержки"
    assert union.get("internal_instruction").display_name == "Внутренние инструкции"
    with pytest.raises(KeyError):
        registries[0].get("internal_instruction")


def test_evidence_rules_apply_to_unclassified_questions_and_restore_scope() -> None:
    support = selected_pack("support").evidence_profile
    internal = selected_pack("internal").evidence_profile
    question = "Расскажи подробнее"
    with evidence_profile_scope(support):
        assert current_evidence_profile().key == "support"
        assert "поддержку" in current_evidence_profile().reading_reminder(question)
        with evidence_profile_scope(internal):
            assert current_evidence_profile().key == "internal"
            assert "согласования" in current_evidence_profile().reading_reminder(question)
        assert current_evidence_profile() is support
    assert support.reading_rules(question) != internal.reading_rules(question)
