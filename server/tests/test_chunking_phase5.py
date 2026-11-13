"""Phase 5 format signals, contract boundaries and safe profile registration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import fitz
from langchain.schema import Document
import pytest

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.ingestion.content_splitter import split_by_content
from domain.ingestion.profiles.catalog import SETTINGS_PROFILE_TYPES
from domain.ingestion.profiles.contract import ContractDomainProfile
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.ingestion.profiles.types import refs_to_metadata
from domain.knowledge.access_control import get_visibility_conditions
from domain.knowledge.enums import DocDomain, DocumentVisibility
from infrastructure.bm25.bm25_index import BM25Index
from infrastructure.domain_profile import register_all_profiles
from infrastructure.domain_profile.answer_policy import classify_query_domain, policy_for
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.ingestion.docx import read_docx
from infrastructure.ml.ingestion.pdf_headings import apply_pdf_headings, toc_headings
from infrastructure.ml.ingestion.registry import FormatContext, pdf_sections
from infrastructure.ml.ingestion.splitting import classify_content_shape
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from infrastructure.repositories.vector.acl import build_qdrant_filter, with_domain_filter
from tests.chunking_fixtures import ChunkingFixture


if TYPE_CHECKING:
    from docx.parts.document import DocumentPart
    from docx.text.paragraph import Paragraph


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


def test_contract_classification_and_law_are_unambiguous(
    profiles: tuple[DomainProfileRegistry, DomainSettingsAdapter],
) -> None:
    registry, settings = profiles
    text = Path("tests/fixtures/chunking/contract.txt").read_text()
    result = registry.classify(text, settings=settings)
    assert result.domain_key == DocDomain.CONTRACT
    assert not result.is_ambiguous
    law = "Закон Республики Беларусь от 28.12.2009 № 113-З\n" + "Закон регулирует отношения. " * 8
    result = registry.classify(law, settings=settings)
    assert result.domain_key == DocDomain.LEGAL
    assert not result.is_ambiguous
    assert registry.get(DocDomain.CONTRACT).resolution_strategy == "replace"


def test_contract_boundaries_preserve_parents_and_appendix() -> None:
    text = Path("tests/fixtures/chunking/contract.txt").read_text()
    profile = ContractDomainProfile()
    units = split_by_content(text, profile.content_boundaries(), 600, unit_label=profile.unit_label)
    assert any(unit.boundary_value == "2.1.1" for unit in units)
    child = next(unit for unit in units if unit.boundary_value == "2.1.1")
    assert any(parent.boundary_value == "2.1" for parent in child.parents)
    assert any(parent.boundary_value == "2" for parent in child.parents)
    assert "1. ПРЕДМЕТ ДОГОВОРА" in "\n".join(unit.content for unit in units) or any(
        parent.heading == "1. ПРЕДМЕТ ДОГОВОРА" for unit in units for parent in unit.parents
    )
    assert any(
        unit.unit_kind == "appendix" or any(p.unit_kind == "appendix" for p in unit.parents) for unit in units
    )


def test_contract_references_do_not_extract_party_names() -> None:
    metadata = refs_to_metadata(
        ContractDomainProfile().extract_references(
            "ДОГОВОР № 17/24 от 1 февраля 2024 г.\nЗаказчик Иванов Иван Иванович. "
            "Исполнитель ООО Секрет. Оплата по п. 3.2, раздел 5, Приложение № 1."
        )
    )
    assert metadata["contract_number"] == "17/24"
    assert metadata["contract_date"] == "1 февраля 2024 г."
    assert metadata["clause"] == "3.2"
    assert metadata["section"] == "5"
    assert metadata["appendix"] == "1"
    assert metadata["party_role"] == ["Заказчик", "Исполнитель"]
    assert "Иванов" not in str(metadata)
    assert "Секрет" not in str(metadata)


@pytest.mark.parametrize(
    "title", ["Закон Республики Беларусь от 28.12.2009 № 113-З", "Федеральный закон от 01.02.2024 № 3-ФЗ"]
)
def test_law_number_and_date(title: str) -> None:
    refs = refs_to_metadata(LegalDomainProfile().extract_references(title))
    assert refs["act_number"] == ("113-З" if "Беларусь" in title else "3-ФЗ")
    assert refs["act_date"] == ("28.12.2009" if "Беларусь" in title else "01.02.2024")


@pytest.mark.parametrize(
    "name",
    [
        "Кодекс Республики Беларусь о браке",
        "Трудовой кодекс Республики Беларусь",
        "Налоговому кодексу Республики Беларусь",
        "Кодекс Республики Беларусь о земле от 23.07.2008 № 425-З",
        "Закона Республики Беларусь от 28.12.2009 № 113-З",
        "ТК",
        "ГК",
        "НК",
        "КоАП",
    ],
)
def test_belarus_law_names(name: str) -> None:
    assert any(
        ref.kind == "law_name" and ref.value.strip() == name
        for ref in LegalDomainProfile().extract_references(name)
    )


def test_pdf_font_headings_midpage_and_page_spans(tmp_path: Path) -> None:
    path = ChunkingFixture("pdf_headings", ".pdf", "general").render(tmp_path)
    parser = LangchainDocumentParser()
    raw = parser.parse(path)
    assert len({doc.metadata.get("section") for doc in raw}) == 5
    assert all(doc.metadata.get("section") for doc in raw)
    chunks = LangchainDocumentSplitter().split(raw)
    assert any(
        c.metadata.get("heading") == "Настройка уведомлений" and c.metadata["pages"] == [2] for c in chunks
    )
    assert all(c.metadata["pages"] in ([1], [2]) for c in chunks)
    assert all(any(hint.get("heading") for hint in d.metadata["line_hints"]) for d in raw)


def test_toc_is_primary_over_font_sizes(tmp_path: Path) -> None:
    path = tmp_path / "toc.pdf"
    with fitz.Document() as pdf:
        page = pdf.new_page()

        for y, text, size in [
            (80, "Root", 11),
            (120, "Body sentence.", 11),
            (160, "Child", 11),
            (200, "Important", 20),
            (230, "More body.", 11),
        ]:
            page.insert_text((72, y), text, fontsize=size)

        pdf.set_toc([[1, "Root", 1], [2, "Child", 1]])
        pdf.save(path)
    parsed = pdf_sections(path, FormatContext())
    assert parsed.structured
    assert [s.heading for s in parsed.sections] == ["Root", "Root > Child"]
    assert all(s.metadata["page"] == 1 for s in parsed.sections)
    assert "Important" in parsed.sections[-1].content


def test_empty_toc_title_does_not_match_blank_line() -> None:
    pages = [Document(page_content="Root\n\nBody.\n\nChild\nMore.", metadata={"page": 1})]

    headings = toc_headings(pages, [[1, "Root", 1], [2, "  ", 1], [2, "Child", 1]])

    assert [(h.level, h.title) for h in headings] == [(1, "Root"), (2, "Child")]


def test_structured_pdf_section_continues_across_pages(tmp_path: Path) -> None:
    path = tmp_path / "pages.pdf"
    with fitz.Document() as pdf:
        for index in range(3):
            page = pdf.new_page()

            if index != 1:
                page.insert_text((72, 80), "First" if index == 0 else "Second", fontsize=18)

            page.insert_text((72, 140), f"Body on page {index + 1} continues with useful information.")

        pdf.save(path)
    chunks = LangchainDocumentSplitter().split(LangchainDocumentParser().parse(path))
    first = next(c for c in chunks if c.metadata.get("section") == "First")
    assert first.metadata["pages"] == [1, 2]
    assert first.metadata["page_start"] == 1
    assert first.metadata["page_end"] == 2
    assert next(c for c in chunks if c.metadata.get("section") == "Second").metadata["pages"] == [3]


def test_plain_pdf_and_single_heading_keep_unstructured_path(tmp_path: Path) -> None:
    path = tmp_path / "plain.pdf"
    with fitz.Document() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 80), "One title", fontsize=18)
        page.insert_text((72, 130), "An ordinary paragraph is still body text.")
        pdf.save(path)
    parsed = pdf_sections(path, FormatContext())
    assert not parsed.structured
    assert len(parsed.sections) == 1
    assert parsed.sections[0].heading is None


def test_ocr_block_titles_and_numbered_headings() -> None:
    with fitz.Document() as pdf:
        pdf.new_page()
        pages = [
            Document(page_content=text, metadata={"page": 1, "content_type": kind, "text_quality": 0})
            for text, kind in [
                ("First section", "title"),
                ("Body text.", "text"),
                ("Second section", "section-header"),
                ("More text.", "text"),
            ]
        ]
        result = apply_pdf_headings(pdf, pages)
        assert {item.metadata["content_type"] for item in result} == {"text"}
        assert [item.metadata["section"] for item in result] == [
            "First section",
            "First section",
            "Second section",
            "Second section",
        ]
        pages = [
            Document(
                page_content="1. General\nBody text.\n2. Procedure\nOther text.",
                metadata={"page": 1, "text_quality": 0},
            )
        ]
        result = apply_pdf_headings(pdf, pages)
        assert [item.metadata["section"] for item in result] == ["1. General", "2. Procedure"]


def test_wrapped_numbered_clause_lines_are_not_pdf_headings() -> None:
    wide = "1.1. Настоящее соглашение заключено в связи с переговорами Сторон о возможной"
    proper = "2.1. Доступ к сведениям предоставляется работникам, которые ознакомлены с"
    content = (
        f"1. ОБЩИЕ ПОЛОЖЕНИЯ\n{wide}\nинтеграции учётной системы с системой EDI-FLOW.\n"
        f"2. ОБЯЗАННОСТИ\n{proper}\nПоложением о конфиденциальности, утверждённым руководителем организации."
    )

    with fitz.Document() as pdf:
        pdf.new_page()
        result = apply_pdf_headings(
            pdf, [Document(page_content=content, metadata={"page": 1, "text_quality": 0})]
        )

    assert [item.metadata["section"] for item in result] == ["1. ОБЩИЕ ПОЛОЖЕНИЯ", "2. ОБЯЗАННОСТИ"]
    assert wide in result[0].page_content
    assert proper in result[1].page_content


def add_numbering(paragraph: Paragraph, level: int) -> None:
    properties = OxmlElement("w:numPr")

    for name, value in [("ilvl", str(level)), ("numId", "5")]:
        element = OxmlElement(f"w:{name}")
        element.set(qn("w:val"), value)
        properties.append(element)

    paragraph._p.get_or_add_pPr().append(properties)
    root = cast("DocumentPart", paragraph.part).numbering_part.element
    num = next(n for n in root.findall(qn("w:num")) if n.get(qn("w:numId")) == "5")
    reference = num.find(qn("w:abstractNumId"))
    assert reference is not None
    abstract = next(
        n
        for n in root.findall(qn("w:abstractNum"))
        if n.get(qn("w:abstractNumId")) == reference.get(qn("w:val"))
    )

    if not any(n.get(qn("w:ilvl")) == str(level) for n in abstract.findall(qn("w:lvl"))):
        definition = OxmlElement("w:lvl")
        definition.set(qn("w:ilvl"), str(level))
        number_format = OxmlElement("w:numFmt")
        number_format.set(qn("w:val"), "decimal")
        definition.append(number_format)
        abstract.append(definition)


def test_docx_numbering_and_bold_create_hierarchy(tmp_path: Path) -> None:
    path = tmp_path / "numbered.docx"
    document = docx.Document()
    general = document.add_paragraph()
    general.add_run("General provisions").bold = True
    add_numbering(general, 0)
    document.add_paragraph("Body with a complete sentence.")
    add_numbering(document.add_paragraph("RESPONSIBILITIES"), 1)
    document.add_paragraph("The nested clause has content.")
    document.add_paragraph().add_run("Final provisions").bold = True
    document.add_paragraph("Closing sentence.")
    document.save(str(path))
    _, sections, _ = read_docx(path)
    assert len(sections) == 3
    assert sections[1].heading
    assert " > " in sections[1].heading
    assert sections[0].metadata["line_hints"][0]["numbering_level"] == 0
    assert sections[1].metadata["line_hints"][0]["numbering_level"] == 1
    assert sections[2].heading == "Final provisions"


def test_plain_numbered_list_items_keep_styled_section(tmp_path: Path) -> None:
    path = tmp_path / "list.docx"
    document = docx.Document()
    document.add_heading("Требования к рабочему месту", 1)
    document.add_paragraph("Необходимо установить следующие программы:")

    for item in ["Microsoft Office", "Google Chrome", "1С Предприятие"]:
        add_numbering(document.add_paragraph(item), 0)

    document.add_paragraph("После установки перезагрузите компьютер.")
    document.save(str(path))
    _, sections, _ = read_docx(path)
    assert [section.heading for section in sections] == ["Требования к рабочему месту"]
    assert "Необходимо установить следующие программы:\n1. Microsoft Office" in sections[0].content


def test_bold_numbered_title_nests_under_styled_heading(tmp_path: Path) -> None:
    path = tmp_path / "nested.docx"
    document = docx.Document()
    document.add_heading("Регламент", 1)
    title = document.add_paragraph()
    title.add_run("Порядок работы").bold = True
    add_numbering(title, 0)
    document.add_paragraph("Текст раздела.")
    document.save(str(path))
    _, sections, _ = read_docx(path)
    assert sections[-1].heading == "Регламент > 1. Порядок работы"


def test_inferred_docx_headings_and_tables_do_not_become_document_title(tmp_path: Path) -> None:
    path = tmp_path / "Инструкция по приёму.docx"
    document = docx.Document()
    document.add_paragraph("Порядок приёма накладной")
    step = document.add_paragraph()
    step.add_run("Вход в систему").bold = True
    add_numbering(step, 0)
    document.add_paragraph("Откройте раздел «Входящие».")
    document.add_table(rows=1, cols=1).cell(0, 0).text = "Поле"
    document.save(str(path))
    docs = LangchainDocumentParser().parse(path)
    assert {doc.metadata["doc_title"] for doc in docs} == {"Инструкция по приёму"}


def test_docx_russian_heading_style(tmp_path: Path) -> None:
    path = tmp_path / "russian.docx"
    document = docx.Document()
    document.styles.add_style("Заголовок 1", WD_STYLE_TYPE.PARAGRAPH)
    document.add_paragraph("Общие положения", "Заголовок 1")
    document.add_paragraph("Содержание раздела.")
    document.save(str(path))
    _, sections, _ = read_docx(path)
    assert sections[0].heading == "Общие положения"
    assert sections[0].content == "Содержание раздела."


def test_format_hint_only_confirms_line_not_invalid_numbering() -> None:
    profile = ContractDomainProfile()
    text = "1. ПРЕДМЕТ\n1.1. Условия без точки\n1.2. Следующий пункт.\n1.1. Повторный номер."
    units = split_by_content(
        text, profile.content_boundaries(), 1000, confirmed_starts=frozenset({text.index("1.2.")})
    )
    assert sum(unit.boundary_value == "1.1" for unit in units) == 1
    assert any(unit.boundary_value == "1.2" for unit in units)


def test_ordinary_prose_is_not_definition_or_procedure() -> None:
    assert classify_content_shape("Уведомление является частью инструкции пользователя.") is None
    assert classify_content_shape("- Первый пункт\n- Второй пункт") == "list"


def test_contract_acl_is_preserved_in_bm25_and_qdrant() -> None:
    documents = [
        Document(
            page_content=f"Условия договора {owner}",
            metadata={
                "doc_domain": DocDomain.CONTRACT.value,
                "owner_id": owner,
                "visibility": DocumentVisibility.INTERNAL_PRIVATE.value,
            },
        )
        for owner in [1, 2]
    ]
    user = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)
    index = BM25Index.from_documents(documents)
    conditions = get_visibility_conditions(UserKind.INTERNAL, [], for_list=False, user_role=UserRole.USER)
    result = index.search_with_hashes(
        "договора", visibility_conditions=conditions, user_id=1, user_group_ids=[]
    )
    assert len(result) == 1
    acl = build_qdrant_filter(user, [])
    combined = with_domain_filter(acl, DocDomain.CONTRACT.value)
    assert isinstance(combined.must, list)
    assert combined.must[0] == acl
    assert '"value":"contract"' in combined.model_dump_json()


@pytest.mark.parametrize("kind", ["УКАЗ", "ДЕКРЕТ"])
def test_belarus_presidential_acts_stay_in_decree(
    profiles: tuple[DomainProfileRegistry, DomainSettingsAdapter],
    kind: str,
) -> None:
    registry, settings = profiles
    text = (
        f"{kind} Президента Республики Беларусь от 01.02.2024 № 3\nПОСТАНОВЛЯЮ:\n1. Утвердить порядок.\n"
        + "Настоящий акт действует со дня опубликования. " * 4
    )
    result = registry.classify(text, settings=settings)
    assert result.domain_key == DocDomain.DECREE
    assert not result.is_ambiguous


def test_decree_reference_line_is_not_a_title() -> None:
    text = "Признать утратившими силу:\nУказ Президента от 1.02.2024 № 3"
    assert DecreeDomainProfile().classify_score(text) == 0


def test_contract_question_uses_contract_citation_policy() -> None:
    assert classify_query_domain("Обязан ли исполнитель по договору вернуть оплату?") == DocDomain.CONTRACT
    policy = policy_for(DocDomain.CONTRACT)
    assert policy.rules is not None
    assert "пункт" in policy.rules
    # Contract questions often need the governing law: retrieval stays on the full ACL scope.
    assert not policy.fallback_to_full_corpus
