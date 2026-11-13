"""Tree round-trips, independent refinement and common packer invariants."""

from dataclasses import replace
import itertools
from pathlib import Path
from typing import Any

import pytest

from domain.ingestion.chunking_policy import ChunkingPolicy
from domain.ingestion.document_tree import DocNode, NodeKind, tree_text, walk_tree
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.general import GeneralDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.text_budget import split_words
from domain.ingestion.tree_packer import ChunkDraft, pack_tree
from domain.knowledge.enums import PageContentType
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.ingestion.amendment_context import attach_table_replacement_scope
from infrastructure.ml.ingestion.format_tree import (
    sections_from_tree,
    tree_from_sections,
    tree_heading_context,
)
from infrastructure.ml.ingestion.sections import ParsedFormat, Section
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser


def test_format_tree_round_trip_preserves_blocks_and_page_metadata() -> None:
    sections = [
        Section("A > B", "First\nparagraph", metadata={"page": 1, "line_hints": [{"start": 0}]}),
        Section("A > B", "| Key | Value |\n|---|---|\n| 1 | one |", PageContentType.TABLE, {"page": 2}),
        Section("A > C", "Second paragraph", metadata={"page": 3}),
        Section(None, "Tail"),
        Section("A > B", "Repeated heading, different node", metadata={"page": 4}),
    ]
    parsed = ParsedFormat(tree_from_sections(sections))
    assert sections_from_tree(parsed.tree) == sections
    source = tree_text(parsed.tree)
    leaves = [node for node in walk_tree(parsed.tree) if "metadata" in node.hints]
    assert all(source[node.span[0] : node.span[1]] == node.text for node in leaves)
    assert len({node.node_id for node in leaves}) == len(leaves)
    assert parsed.tree.pages == (1, 2, 3, 4)


def test_heading_labels_with_separator_are_not_parsed_as_breadcrumbs() -> None:
    body = DocNode(
        NodeKind.PARAGRAPH, text="Body", span=(0, 4), hints={"metadata": {}, "heading_path": ("A > B",)}
    )
    heading = DocNode(
        NodeKind.HEADING_BLOCK,
        label="A > B",
        children=(body,),
        span=(0, 4),
        hints={"heading_path": ("A > B",)},
    )
    tree = DocNode(NodeKind.DOCUMENT, children=(heading,), span=(0, 4), hints={"source_text": "Body"})
    context = tree_heading_context(tree)
    assert context["level"]["A > B"] == 1
    assert context["immediate"]["A > B"] == "A > B"
    docs = LangchainDocumentParser.tree_to_documents(tree, Path("doc.md"))
    assert docs[0].metadata["heading_level"] == 1
    assert docs[0].page_content == "Body"


def test_profiles_refine_before_budgeting_and_general_is_identity() -> None:
    text = "Статья 1. Правила\n1. Условия:\n1.1. Первое правило.\n1.2. Второе правило."
    tree = DocNode(NodeKind.DOCUMENT, text=text, span=(0, len(text)), hints={"source_text": text})
    assert GeneralDomainProfile().refine(tree) is tree
    refined = LegalDomainProfile().refine(tree)
    nodes = walk_tree(refined)
    assert {NodeKind.ARTICLE, NodeKind.POINT, NodeKind.SUBPOINT_NUM} <= {node.kind for node in nodes}
    assert tree_text(refined) == text
    assert all(text[node.span[0] : node.span[1]] == tree_text(node) for node in nodes)


def test_profile_policy_reads_its_own_scope_and_rejects_invalid_pair() -> None:
    settings = DomainSettingsAdapter()
    settings.replace_snapshot(
        {("legal_chunk_target_tokens", "legal"): "120", ("legal_chunk_max_tokens", "legal"): "200"}
    )
    policy = LegalDomainProfile().chunking_policy(settings)
    assert (policy.target_tokens, policy.max_tokens, policy.overlap_tokens) == (120, 200, 0)
    assert DecreeDomainProfile().chunking_policy(settings).max_tokens == 800
    settings.replace_snapshot({("legal_chunk_target_tokens", "legal"): "900"})
    with pytest.raises(ValueError, match="Target"):
        LegalDomainProfile().chunking_policy(settings)


def test_overlap_is_profile_data_not_domain_name_check() -> None:
    settings = DomainSettingsAdapter()
    settings.replace_snapshot(
        {
            ("general_chunk_overlap_tokens", "general"): "30",
            ("legal_chunk_overlap_tokens", "legal"): "30",
        }
    )
    assert GeneralDomainProfile().chunking_policy(None).overlap_tokens == 50
    assert GeneralDomainProfile().chunking_policy(settings).overlap_tokens == 30
    # Structured profiles forbid overlap even when a stray setting exists.
    assert LegalDomainProfile().chunking_policy(settings).overlap_tokens == 0
    assert DecreeDomainProfile().chunking_policy(settings).overlap_tokens == 0


def test_refine_reports_boundary_events_without_storing_callbacks_in_tree() -> None:
    text = "Статья 1. Правила\n1. Первое.\n3. Пропуск номера."
    tree = DocNode(NodeKind.DOCUMENT, text=text, span=(0, len(text)), hints={"source_text": text})
    events: list[tuple[str, int]] = []
    refined = LegalDomainProfile().refine(tree, lambda rule, pos: events.append((rule, pos)))
    assert events
    assert all(0 <= pos < len(text) for _, pos in events)
    assert not any(callable(value) for node in walk_tree(refined) for value in node.hints.values())


@pytest.mark.parametrize(
    "values",
    [
        {"target_tokens": 0},
        {"max_tokens": 20},
        {"overlap_tokens": 50},
        {"context_depth": -1},
        {"min_tokens": -1},
    ],
)
def test_policy_validation(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"Target|Overlap|Minimum"):
        replace(ChunkingPolicy(50, 100), **values)


def test_step_atoms_pack_without_losing_source_spans_or_numbers() -> None:
    source = "1. First step.\n2. Second step.\n3. Third step."
    starts = [0, source.index("2."), source.index("3."), len(source)]
    children = tuple(
        DocNode(NodeKind.STEP, number=str(i + 1), text=source[start:end], span=(start, end), pages=(i + 1,))
        for i, (start, end) in enumerate(itertools.pairwise(starts))
    )
    tree = DocNode(NodeKind.DOCUMENT, children=children, span=(0, len(source)), hints={"source_text": source})
    drafts = pack_tree(tree, ChunkingPolicy(31, 35), len)
    assert [draft.numbers for draft in drafts] == [("1", "2"), ("3",)]
    assert all(source[draft.span[0] : draft.span[1]] == draft.content for draft in drafts)
    assert [draft.pages for draft in drafts] == [(1, 2), (3,)]
    separate = pack_tree(tree, ChunkingPolicy(31, 35, pack_siblings=False), len)
    assert [draft.numbers for draft in separate] == [("1",), ("2",), ("3",)]


def test_oversized_leaf_falls_back_and_retains_repeated_source_offsets() -> None:
    source = "Same sentence. " * 7 + "x" * 100
    tree = DocNode(NodeKind.PARAGRAPH, text=source, span=(0, len(source)), hints={"source_text": source})
    fallbacks = []
    drafts = pack_tree(
        tree, ChunkingPolicy(35, 40, overlap_tokens=15), len, on_fallback=lambda: fallbacks.append(True)
    )
    assert fallbacks
    assert all(len(draft.content) <= 40 for draft in drafts)
    assert all(source[draft.span[0] : draft.span[1]] == draft.content for draft in drafts)
    assert all(left.span[0] < right.span[0] for left, right in itertools.pairwise(drafts))
    assert "".join(draft.content for draft in drafts).count("x") == 100


def test_measured_fallback_keeps_special_token_cost_at_empty_joins() -> None:
    def measure(text: str) -> int:
        return len(text) + 2

    pieces = split_words("one two three four five six", 18, measure)
    assert pieces == ["one two", "three four", "five six"]


def test_dense_policy_excludes_title_without_changing_body() -> None:
    tree = DocNode(
        NodeKind.PARAGRAPH,
        text="Body text.",
        span=(0, 10),
        hints={"source_text": "Body text.", "metadata": {"doc_title": "Important title"}},
    )
    enabled = pack_tree(tree, ChunkingPolicy(100, 150), len)[0]
    disabled = pack_tree(tree, ChunkingPolicy(100, 150, include_doc_title=False), len)[0]
    assert enabled.content == disabled.content == "Body text."
    assert "Important title" in enabled.embed_text
    assert "Important title" not in disabled.embed_text


def test_refined_tree_preview_does_not_duplicate_format_blocks() -> None:
    parsed = ParsedFormat(tree_from_sections([Section("Правила", "Статья 1. Права\n1. Первое.\n2. Второе.")]))
    refined = LegalDomainProfile().refine(parsed.tree)
    assert sections_from_tree(refined) == parsed.sections


@pytest.mark.parametrize("same_scope", [True, False])
def test_table_replacements_use_tree_parent_identity(same_scope: bool) -> None:
    table = "| Код | Значение |\n|---|---|\n| A | значение |"
    tree = tree_from_sections(
        [
            Section("A", "В таблице 1 позицию"),
            Section("A", table, PageContentType.TABLE, {"content_type": "table"}),
            Section("A", "заменить позицией"),
            Section("A" if same_scope else "B", table, PageContentType.TABLE, {"content_type": "table"}),
        ]
    )
    attach_table_replacement_scope(tree)
    tables = [node for node in walk_tree(tree) if node.kind == NodeKind.TABLE]
    assert all(bool(node.hints["metadata"].get("parent_units")) == same_scope for node in tables)


def test_refinement_maps_child_pages_from_source_offsets() -> None:
    first = "Статья 1. Первая\nТекст первой.\n"
    second = "Статья 2. Вторая\nТекст второй."
    source = first + second
    tree = DocNode(
        NodeKind.DOCUMENT,
        text=source,
        span=(0, len(source)),
        pages=(1, 2),
        hints={
            "source_text": source,
            "page_spans": [(0, len(first), 1), (len(first), len(source), 2)],
        },
    )
    articles = [
        node for node in walk_tree(LegalDomainProfile().refine(tree)) if node.kind == NodeKind.ARTICLE
    ]
    assert [node.pages for node in articles] == [(1,), (2,)]


def test_mixed_tree_dispatches_tables_without_packing_across_them() -> None:
    tree = tree_from_sections(
        [
            Section("A", "Introduction."),
            Section("A", "| Code |\n|---|\n| A |", PageContentType.TABLE),
            Section("A", "Following text."),
        ]
    )
    visited = []

    def table_packer(node: DocNode) -> list[ChunkDraft]:
        visited.append(node)
        return pack_tree(node, ChunkingPolicy(100, 150), len)

    drafts = pack_tree(tree, ChunkingPolicy(100, 150), len, table_packer=table_packer)
    assert len(visited) == 1
    assert [draft.unit_kind for draft in drafts] == [NodeKind.PARAGRAPH, NodeKind.TABLE, NodeKind.PARAGRAPH]
    assert [draft.content for draft in drafts] == [section.content for section in sections_from_tree(tree)]
