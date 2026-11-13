"""Format parsers build the document tree directly (6.1b).

Hierarchy comes from heading levels, spans index one normalized document text,
and chunk-facing output equals the former breadcrumb bridge for every format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docx import Document as WordDocument
import pytest

from domain.ingestion.document_tree import NodeKind, tree_text, walk_tree
from infrastructure.ml.ingestion.format_tree import (
    DocumentTreeBuilder,
    content_blocks,
    sections_from_tree,
    tree_from_sections,
    tree_heading_context,
)
from infrastructure.ml.ingestion.registry import FORMAT_HANDLERS, FormatContext
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser
from tests.chunking_fixtures import ChunkingFixture


if TYPE_CHECKING:
    from pathlib import Path

    from domain.ingestion.document_tree import DocNode


def assert_spans_index_source(tree: DocNode) -> None:
    source = tree_text(tree)
    assert tree.span == (0, len(source))

    for node in walk_tree(tree):
        assert source[node.span[0] : node.span[1]] == tree_text(node)


def test_builder_nests_by_level_and_emits_only_labels_outside_blocks() -> None:
    builder = DocumentTreeBuilder()
    builder.block("Preamble.")
    builder.heading(1, "Chapter > One")
    builder.heading(2, "1. Inferred title", emit=False)
    builder.block("1. Inferred title\nBody.")
    builder.heading(1, "Chapter Two")
    builder.block("Tail.", metadata={"page": 3})
    tree = builder.build()

    assert_spans_index_source(tree)
    assert tree_text(tree) == "Preamble.\n\nChapter > One\n\n1. Inferred title\nBody.\n\nChapter Two\n\nTail."
    paths = [block.hints["heading_path"] for block in content_blocks(tree)]
    assert paths == [(), ("Chapter > One", "1. Inferred title"), ("Chapter Two",)]
    # A separator inside a label is not a breadcrumb level.
    assert tree_heading_context(tree)["level"]["Chapter > One > 1. Inferred title"] == 2
    assert tree.hints["page_spans"] == [(len(tree_text(tree)) - len("Tail."), len(tree_text(tree)), 3)]
    assert tree.pages == (3,)


def test_headings_without_blocks_stay_in_tree_but_not_in_navigation() -> None:
    builder = DocumentTreeBuilder()
    builder.heading(1, "Empty")
    builder.heading(1, "Filled")
    builder.block("Body.")
    builder.heading(1, "Filled")
    builder.block("Second body under a repeated label.")
    tree = builder.build()

    assert_spans_index_source(tree)
    assert [node.label for node in tree.children] == ["Empty", "Filled", "Filled"]
    context = tree_heading_context(tree)
    assert "Empty" not in context["level"]
    assert context["toc"] == ["Filled"]
    assert context["siblings"]["Filled"] == []


def write_rtf(path: Path) -> None:
    path.write_text(
        r"{\rtf1\ansi\fs22 "
        r"{\pard\b\fs32 Раздел 1 Общие положения\par}"
        r"{\pard Настоящий порядок применяется ко всем заявлениям граждан.\par}"
        r"{\pard\b\fs28 Глава 2 Подача заявления\par}"
        r"{\pard Заявление подаётся лично или по почте в течение месяца.\par}"
        r"{\pard\b\fs32 Раздел 3 Заключительные положения\par}"
        r"{\pard Порядок вступает в силу после официального опубликования.\par}"
        r"}",
        encoding="utf-8",
    )


def write_styled_docx(path: Path) -> None:
    document = WordDocument()
    document.add_heading("Регламент", level=1)
    document.add_paragraph("Общие положения регламента.")
    document.add_heading("Порядок работы", level=2)
    document.add_paragraph("Сотрудник открывает карточку документа.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Поле"
    table.cell(0, 1).text = "Значение"
    table.cell(1, 0).text = "Срок"
    table.cell(1, 1).text = "5 дней"
    document.add_heading("Контроль", level=1)
    document.add_paragraph("Руководитель проверяет исполнение.")
    document.save(str(path))


FIXTURES = [
    ChunkingFixture("law", ".md", "legal"),
    ChunkingFixture("pdf_headings", ".md", "general"),
    ChunkingFixture("instruction", ".docx", "general"),
    ChunkingFixture("contract", ".docx", "contract"),
    ChunkingFixture("table", ".docx", "general"),
    ChunkingFixture("pdf_headings", ".pdf", "general"),
    ChunkingFixture("pdf_hard_wraps", ".pdf", "general"),
    ChunkingFixture("instruction", ".txt", "general"),
]


def rendered_documents(tmp_path: Path) -> list[Path]:
    paths = [fixture.render(tmp_path) for fixture in FIXTURES]
    write_rtf(tmp_path / "rules.rtf")
    write_styled_docx(tmp_path / "styled.docx")
    return [*paths, tmp_path / "rules.rtf", tmp_path / "styled.docx"]


@pytest.mark.parametrize("index", range(len(FIXTURES) + 2))
def test_parser_trees_match_breadcrumb_bridge_and_index_one_source(tmp_path: Path, index: int) -> None:
    path = rendered_documents(tmp_path)[index]
    parsed = FORMAT_HANDLERS[path.suffix].parse(path, FormatContext())

    assert_spans_index_source(parsed.tree)
    bridged = tree_from_sections(sections_from_tree(parsed.tree))
    assert sections_from_tree(bridged) == sections_from_tree(parsed.tree)
    assert tree_heading_context(parsed.tree) == tree_heading_context(bridged)

    if parsed.structured:
        direct = LangchainDocumentParser.tree_to_documents(parsed.tree, path, parsed.metadata)
        legacy = LangchainDocumentParser.tree_to_documents(bridged, path, parsed.metadata)
        assert [(doc.page_content, doc.metadata) for doc in direct] == [
            (doc.page_content, doc.metadata) for doc in legacy
        ]


def test_parsers_report_real_hierarchy(tmp_path: Path) -> None:
    write_rtf(tmp_path / "rules.rtf")
    write_styled_docx(tmp_path / "styled.docx")
    rtf = FORMAT_HANDLERS[".rtf"].parse(tmp_path / "rules.rtf", FormatContext())
    docx = FORMAT_HANDLERS[".docx"].parse(tmp_path / "styled.docx", FormatContext())

    assert [block.hints["heading_path"] for block in content_blocks(rtf.tree)] == [
        ("Раздел 1 Общие положения",),
        ("Раздел 1 Общие положения", "Глава 2 Подача заявления"),
        ("Раздел 3 Заключительные положения",),
    ]
    docx_paths = [(block.kind, block.hints["heading_path"]) for block in content_blocks(docx.tree)]
    assert docx_paths == [
        (NodeKind.PARAGRAPH, ("Регламент",)),
        (NodeKind.PARAGRAPH, ("Регламент", "Порядок работы")),
        (NodeKind.TABLE, ("Регламент", "Порядок работы")),
        (NodeKind.PARAGRAPH, ("Контроль",)),
    ]
    # Styled DOCX headings leave the body, so the heading node carries its own label.
    source = tree_text(docx.tree)
    assert source.startswith("Регламент\n\nОбщие положения регламента.")


def test_pdf_tree_maps_pages_from_source_offsets(tmp_path: Path) -> None:
    path = ChunkingFixture("pdf_headings", ".pdf", "general").render(tmp_path)
    tree = FORMAT_HANDLERS[".pdf"].parse(path, FormatContext()).tree
    source = tree_text(tree)
    spans = tree.hints["page_spans"]

    assert spans
    assert all(low < high <= len(source) for low, high, _page in spans)

    for block in content_blocks(tree):
        assert block.pages == tuple(page for low, high, page in spans if (low, high) == block.span)

    for node in walk_tree(tree):
        assert set(node.pages) == {page for block in content_blocks(node) for page in block.pages}
