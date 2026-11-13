"""5.6: appendices of one source are classified separately, only for splitting."""

from __future__ import annotations

import pytest

from domain.ingestion.appendix import appendix_cuts, appendix_policy, appendix_starts, classify_appendix
from domain.ingestion.chunking_policy import ChunkingPolicy
from domain.ingestion.profiles.catalog import SETTINGS_PROFILE_TYPES
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.chunk import extract_chunk_context
from domain.knowledge.enums import DocDomain
from infrastructure.domain_profile import register_all_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.langchain_document_parser import LangchainDocumentSplitter


BODY = (
    "1. Утвердить Инструкцию о порядке представления сведений (прилагается).\n"
    "2. Установить форму уведомления согласно приложению 1.\n"
    "3. Настоящее постановление вступает в силу после его официального опубликования."
)
FORM = (
    "Приложение 1\nк постановлению\nМинистерства по налогам и сборам\n"
    "Форма\nУВЕДОМЛЕНИЕ\nоб отнесении к недействительным контрольных знаков\n"
    + "Наименование субъекта хозяйствования ______________________________\n"
    * 4
)
INSTRUCTION = "УТВЕРЖДЕНО\nПостановление\nМинистерства по налогам и сборам\nИНСТРУКЦИЯ\n" + "\n".join(
    f"{n}. Субъект хозяйствования представляет сведения о реализованных товарах в срок до {n} числа."
    for n in range(1, 9)
)


@pytest.fixture
def profiles() -> tuple[DomainProfileRegistry, DomainSettingsAdapter]:
    values = {
        (item.key, cls.key): item.value for cls in SETTINGS_PROFILE_TYPES for item in cls().config_defaults()
    }
    settings = DomainSettingsAdapter()
    settings.replace_snapshot(values)
    registry = DomainProfileRegistry()
    register_all_profiles(registry, settings)
    return registry, settings


def test_appendix_markers_ignore_attachment_notes_and_form_approvals() -> None:
    text = (
        "Приложение 2\nк постановлению\n"
        "Приложение к Инструкции о порядке хранения\n"
        "Приложение № 1\nк договору поставки\n"
        "УТВЕРЖДЕНО\nПостановление Министерства\n"
        "Приложение: на 2 л.\n"
        "УТВЕРЖДАЮ\nРуководитель\n"
        "согласно приложению 1 к постановлению\n"
    )

    assert len(appendix_starts(text)) == 4


def test_appendix_cuts_start_at_block_marker_or_after_short_tail() -> None:
    blocks = [
        "Приложение 1\nк договору\nшапка документа",  # the document itself, not an appendix
        BODY,
        FORM,
        "Глава 3\nТекст главы.\nПриложение 2\nк Инструкции о порядке",  # attribution only
        "УВЕДОМЛЕНИЕ",
        BODY + "\n" + INSTRUCTION + "\n" + "Текст инструкции. " * 60,
    ]

    cuts = appendix_cuts(blocks)

    assert cuts == [(2, 0), (4, 0), (5, len(BODY) + 1)]


def test_appendix_inherits_document_domain_on_weak_signal(
    profiles: tuple[DomainProfileRegistry, DomainSettingsAdapter],
) -> None:
    registry, settings = profiles
    decree = DocDomain.DECREE.value

    assert classify_appendix(registry, "Форма", document_domain=decree, settings=settings) == decree
    assert classify_appendix(registry, INSTRUCTION, document_domain=decree, settings=settings) == decree
    assert (
        classify_appendix(registry, FORM, document_domain=decree, settings=settings)
        == DocDomain.GENERAL.value
    )


def test_appendix_keeps_document_budget_and_forbidden_overlap() -> None:
    document = ChunkingPolicy(target_tokens=400, max_tokens=800)
    general = ChunkingPolicy(target_tokens=350, max_tokens=512, overlap_tokens=50)

    general_in_decree = appendix_policy(general, document)
    decree_in_general = appendix_policy(document, general)

    assert (general_in_decree.target_tokens, general_in_decree.max_tokens) == (400, 800)
    assert general_in_decree.overlap_tokens == 50
    assert (decree_in_general.target_tokens, decree_in_general.max_tokens) == (350, 512)
    assert decree_in_general.overlap_tokens == 0


def section_docs() -> list[RawDocument]:
    return [
        RawDocument(text, {"source": "resolution.docx", "section": section})
        for section, text in (("ПОСТАНОВЛЕНИЕ", BODY), ("Форма", FORM), ("ИНСТРУКЦИЯ", INSTRUCTION))
    ]


def test_appendix_profile_splits_only_its_root(
    profiles: tuple[DomainProfileRegistry, DomainSettingsAdapter],
) -> None:
    registry, settings = profiles

    chunks = LangchainDocumentSplitter(registry, settings).split(section_docs(), DocDomain.DECREE.value)

    form = [c for c in chunks if "УВЕДОМЛЕНИЕ" in c.page_content]
    others = [c for c in chunks if c not in form]
    assert form
    assert all(c.metadata["unit_domain"] == DocDomain.GENERAL.value for c in form)
    assert all("unit_domain" not in c.metadata for c in others)
    assert all("вступает в силу" not in c.page_content for c in form)
    # The approved instruction keeps the document profile and its point boundaries.
    assert any(c.metadata.get("unit_kind") == "point" and "до 8 числа" in c.page_content for c in others)
    assert all("doc_domain" not in c.metadata for c in chunks)
    assert extract_chunk_context(form[0].metadata)["unit_domain"] == DocDomain.GENERAL.value


def test_without_registry_document_profile_splits_everything() -> None:
    chunks = LangchainDocumentSplitter().split(section_docs(), DocDomain.DECREE.value)

    assert chunks
    assert all("unit_domain" not in c.metadata for c in chunks)


def test_long_appendix_inside_single_block_is_cut(
    profiles: tuple[DomainProfileRegistry, DomainSettingsAdapter],
) -> None:
    registry, settings = profiles
    form = FORM + "Подпись руководителя ____________ Дата ____________\n" * 20
    text = BODY + "\n" + form

    chunks = LangchainDocumentSplitter(registry, settings).split(
        [RawDocument(text, {"source": "resolution.txt"})], DocDomain.DECREE.value
    )

    body = [c for c in chunks if "вступает в силу" in c.page_content]
    appendix = [c for c in chunks if "Подпись руководителя" in c.page_content]
    assert body
    assert appendix
    assert all("Подпись руководителя" not in c.page_content for c in body)
    assert all(c.metadata["unit_domain"] == DocDomain.GENERAL.value for c in appendix)


def test_bare_heading_before_its_table_is_not_indexed() -> None:
    def block(text: str, section: str, *, table: bool = False) -> RawDocument:
        meta = {"source": "a.docx", "section": section, "heading": section}
        return RawDocument(text, {**meta, "content_type": "table"} if table else meta)

    docs = [
        block("ТАРИФЫ", "ТАРИФЫ"),
        block("| Услуга | Цена |\n|---|---|\n| Подключение | 100 |", "ТАРИФЫ", table=True),
        block("ПОРЯДОК", "ПОРЯДОК"),
        block("Оплата производится ежемесячно.", "ПОРЯДОК"),
        block("ИТОГИ", "ИТОГИ"),
        block("| Итог | 100 |\n|---|---|", "ДРУГОЙ РАЗДЕЛ", table=True),
    ]

    chunks = LangchainDocumentSplitter().split(docs, DocDomain.CONTRACT.value)

    assert not any(c.page_content.strip() == "ТАРИФЫ" for c in chunks)
    assert any("Подключение" in c.page_content and c.metadata["section"] == "ТАРИФЫ" for c in chunks)
    # A heading with its own text, or before a table of another section, stays.
    assert any("ПОРЯДОК" in c.page_content or c.metadata.get("section") == "ПОРЯДОК" for c in chunks)
    assert any(c.page_content.strip() == "ИТОГИ" for c in chunks)


def test_bare_attribution_becomes_context_of_its_table() -> None:
    docs = [
        RawDocument("1.1. Исполнитель оказывает услуги по тарифам.", {"source": "c.docx", "section": "1"}),
        RawDocument(
            "Приложение № 1\nк договору № 17-EDI/2026",
            {"source": "c.docx", "section": "Приложение № 1", "heading": "Приложение № 1"},
        ),
        RawDocument(
            "| Услуга | Цена |\n|---|---|\n| Подключение | 100 |",
            {"source": "c.docx", "section": "ТАРИФЫ", "heading": "ТАРИФЫ", "content_type": "table"},
        ),
    ]

    chunks = LangchainDocumentSplitter().split(docs, DocDomain.CONTRACT.value)

    assert not any(c.page_content.startswith("Приложение № 1") for c in chunks)
    table = next(c for c in chunks if "Подключение" in c.page_content)
    assert any(
        unit["unit_kind"] == "table_context" and unit["content"].startswith("Приложение № 1")
        for unit in table.metadata["parent_units"]
    )
    assert table.embed_text is not None
    assert "Приложение № 1" in table.embed_text
