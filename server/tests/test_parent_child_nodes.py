"""6.4: stable parent-node ids of chunks and their persisted scopes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import pytest

from application import indexing
from domain.ingestion.profiles.catalog import SETTINGS_PROFILE_TYPES
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.chunk import extract_chunk_context
from domain.knowledge.document_nodes import (
    NODE_PATH_KEY,
    PARENT_NODE_ID_KEY,
    DocumentNode,
    NodeSegment,
    assign_chunk_nodes,
    assign_node_paths,
    build_document_nodes,
    measured_chain,
    node_ancestry,
    node_segments,
    select_parent,
    within_chunk_limit,
)
from domain.knowledge.enums import DocDomain, DocumentVisibility
from infrastructure.domain_profile import register_all_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from tests.chunking_fixtures import FIXTURES_DIR
from tests.test_chunking_appendices import BODY, INSTRUCTION


if TYPE_CHECKING:
    from pathlib import Path


ARTICLE_1 = NodeSegment("article", "Статья 1. Термины", "1")
ARTICLE_2 = NodeSegment("article", "Статья 2. Сфера", "2")
POINT_1 = NodeSegment("point", "1.", "1")


def test_repeated_tokens_get_occurrence_suffix_only_after_interruption() -> None:
    paths = assign_node_paths(
        [
            [ARTICLE_1],
            [ARTICLE_1, POINT_1],
            [ARTICLE_1, POINT_1],  # continuation of the same point
            [ARTICLE_1],  # text directly in the article closes the point
            [ARTICLE_1, POINT_1],  # a new point "1." (for example, of a nested list)
            [ARTICLE_2, POINT_1],
            [ARTICLE_1],  # article number repeated later (appendix without root marker)
        ]
    )

    assert paths == [
        ("article:1",),
        ("article:1", "point:1"),
        ("article:1", "point:1"),
        ("article:1",),
        ("article:1", "point:1#2"),
        ("article:2", "point:1"),
        ("article:1#2",),
    ]


def test_segments_join_style_heading_with_its_unit_and_skip_context_only_parents() -> None:
    metadata = {
        "heading_path": ["Глава 1", "Статья 1. Термины"],
        "parent_units": [
            {"heading": "Введение", "unit_kind": "preamble", "boundary_value": None, "content": "..."},
            {"heading": "Статья 1. Термины", "unit_kind": "article", "boundary_value": "1", "content": ""},
            {"heading": "1.", "unit_kind": "point", "boundary_value": "1", "content": ""},
            {"heading": "Таблица", "unit_kind": "table_context", "boundary_value": None, "content": "x"},
        ],
    }
    appendix = NodeSegment("appendix", "Приложение 1", "1")

    segments = node_segments(metadata, appendix=appendix)

    assert [(segment.kind, segment.number) for segment in segments] == [
        ("appendix", "1"),
        ("heading", None),
        ("heading", None),
        ("point", "1"),
    ]


def test_number_tokens_escape_the_separator() -> None:
    assert NodeSegment("point", "5/1", "5/1").token == "point:5%2F1"  # noqa: S105 - node id, not a secret
    assert NodeSegment("heading", "Раздел").token == NodeSegment("heading", "  раздел ").token


def test_document_nodes_cover_descendant_chunk_ranges() -> None:
    metadata: list[dict[str, Any]] = [{"parent_units": []}]
    metadata += [{"parent_units": [unit("article", "1")]}] * 2
    metadata += [{"parent_units": [unit("article", "1"), unit("point", "2")], "pages": [3, 4]}]
    metadata += [{"parent_units": [unit("article", "2")], "page": 5}]
    assigned = assign_chunk_nodes([(item, None) for item in metadata])
    chunks = [
        (f"text {i}", {**item, **node}) for i, (item, node) in enumerate(zip(metadata, assigned, strict=True))
    ]

    nodes = {node.node_id: node for node in build_document_nodes(chunks)}

    assert assigned[0][PARENT_NODE_ID_KEY] is None
    assert assigned[3][PARENT_NODE_ID_KEY] == "article:1/point:2"
    assert set(nodes) == {"article:1", "article:1/point:2", "article:2"}
    assert (nodes["article:1"].chunk_start, nodes["article:1"].chunk_end) == (1, 3)
    assert nodes["article:1"].char_count == 3 * len("text 1")
    assert nodes["article:1"].pages == (3, 4)
    assert nodes["article:1/point:2"].parent_id == "article:1"
    assert nodes["article:1/point:2"].depth == 2
    assert nodes["article:2"].pages == (5,)


def test_parent_selection_climbs_only_through_fitting_ancestors() -> None:
    chain = [node("a/b/c", 100), node("a/b", 300), node("a", 5000)]

    assert node_ancestry("a/b/c") == ["a/b/c", "a/b", "a"]
    assert select_parent(chain, 400) == chain[1]
    assert select_parent(chain, 50) is None
    assert select_parent([node("a/b/c", 100), node("a/b", 9000), node("a", 300)], 400) == chain[0]


@pytest.fixture(scope="module")
def ingestion() -> tuple[LangchainDocumentParser, LangchainDocumentSplitter]:
    settings = DomainSettingsAdapter()
    settings.replace_snapshot(
        {
            (item.key, cls.key): item.value
            for cls in SETTINGS_PROFILE_TYPES
            for item in cls().config_defaults()
        }
    )
    registry = DomainProfileRegistry()
    register_all_profiles(registry, settings)
    return LangchainDocumentParser(registry, settings), LangchainDocumentSplitter(registry, settings)


def split_text(
    ingestion: tuple[LangchainDocumentParser, LangchainDocumentSplitter], path: Path, domain: str
) -> list[RawDocument]:
    parser, splitter = ingestion
    return splitter.split(parser.parse(path), domain=domain)


def test_resplitting_the_same_file_yields_the_same_node_ids(
    ingestion: tuple[LangchainDocumentParser, LangchainDocumentSplitter], tmp_path: Path
) -> None:
    path = tmp_path / "law.txt"
    path.write_text((FIXTURES_DIR / "law.txt").read_text(encoding="utf-8"), encoding="utf-8")

    first = split_text(ingestion, path, DocDomain.LEGAL.value)
    second = split_text(ingestion, path, DocDomain.LEGAL.value)
    ids = [chunk.metadata[PARENT_NODE_ID_KEY] for chunk in first]

    assert ids == [chunk.metadata[PARENT_NODE_ID_KEY] for chunk in second]
    # Articles fit the budget, so their chunks hang directly under the chapters.
    assert {node_id for node_id in ids if node_id} == {"chapter:1", "chapter:2", "chapter:3"}
    assert all("heading_path" not in chunk.metadata for chunk in first)
    # Every chunk parent is a stored node with a range covering the chunk.
    nodes = {
        item.node_id: item for item in build_document_nodes([(c.page_content, c.metadata) for c in first])
    }

    for index, node_id in enumerate(ids):
        if node_id:
            assert nodes[node_id].chunk_start <= index <= nodes[node_id].chunk_end


def test_node_ids_do_not_depend_on_text_offsets(
    ingestion: tuple[LangchainDocumentParser, LangchainDocumentSplitter], tmp_path: Path
) -> None:
    source = (FIXTURES_DIR / "law.txt").read_text(encoding="utf-8")
    preamble = "Настоящий Закон определяет правовые основы"
    shifted = source.replace(preamble, "Принят в целях развития оборота. " + preamble, 1)
    assert shifted != source
    original_path, shifted_path = tmp_path / "law.txt", tmp_path / "law-shifted.txt"
    original_path.write_text(source, encoding="utf-8")
    shifted_path.write_text(shifted, encoding="utf-8")

    def ids_by_body(path: Path) -> dict[str, Any]:
        chunks = split_text(ingestion, path, DocDomain.LEGAL.value)
        return {chunk.page_content: chunk.metadata[PARENT_NODE_ID_KEY] for chunk in chunks}

    original, moved = ids_by_body(original_path), ids_by_body(shifted_path)
    common = original.keys() & moved.keys()

    assert len(common) > 3
    assert all(original[body] == moved[body] for body in common)


def test_appendix_root_is_a_separate_top_level_scope() -> None:
    settings = DomainSettingsAdapter()
    settings.replace_snapshot(
        {
            (item.key, cls.key): item.value
            for cls in SETTINGS_PROFILE_TYPES
            for item in cls().config_defaults()
        }
    )
    registry = DomainProfileRegistry()
    register_all_profiles(registry, settings)
    docs = [
        RawDocument(BODY, {"source": "resolution.docx", "heading_path": ["ПОСТАНОВЛЕНИЕ"]}),
        RawDocument(INSTRUCTION, {"source": "resolution.docx", "heading_path": ["ИНСТРУКЦИЯ"]}),
    ]

    chunks = LangchainDocumentSplitter(registry, settings).split(docs, DocDomain.DECREE.value)
    ids = [chunk.metadata[PARENT_NODE_ID_KEY] or "" for chunk in chunks]
    instruction = [i for i, chunk in enumerate(chunks) if "до 8 числа" in chunk.page_content]

    assert instruction
    assert all(ids[i].startswith("appendix:1/") for i in instruction)
    body = [i for i, chunk in enumerate(chunks) if "вступает в силу" in chunk.page_content]
    assert body
    assert not any(ids[i].startswith("appendix:") for i in body)


@pytest.mark.asyncio
async def test_pipeline_replaces_nodes_with_chunks_and_drops_transient_path() -> None:
    metadata = [{"parent_units": [unit("article", "1")]}, {"parent_units": [unit("article", "2")]}]
    chunks = [
        RawDocument(f"body {i}", {**item, **node, "act_version_id": 5})
        for i, (item, node) in enumerate(
            zip(metadata, assign_chunk_nodes([(m, None) for m in metadata]), strict=True)
        )
    ]
    uow = Mock()
    uow.chunks.get_ids_by_document_id = AsyncMock(return_value=[])
    uow.chunks.bulk_insert = AsyncMock(return_value=[10, 11])
    uow.chunks.replace_document_nodes = AsyncMock()
    uow.vector_outbox.enqueue = AsyncMock()
    uow.documents.update_status = AsyncMock()

    await indexing.process_chunks_in_uow(
        uow,
        9,
        "law.docx",
        chunks,
        DocumentVisibility.INTERNAL_PUBLIC.value,
        owner_id=None,
        group_id=None,
        doc_domain="legal",
    )

    document_id, act_version_id, nodes = uow.chunks.replace_document_nodes.await_args.args
    assert (document_id, act_version_id) == (9, 5)
    assert [(item.node_id, item.chunk_start) for item in nodes] == [("article:1", 0), ("article:2", 1)]
    inserted = uow.chunks.bulk_insert.await_args.kwargs["chunks"]
    assert [item.context_metadata[PARENT_NODE_ID_KEY] for item in inserted] == ["article:1", "article:2"]
    assert all(NODE_PATH_KEY not in item.context_metadata for item in inserted)
    points = uow.vector_outbox.enqueue.await_args.args[0].payload["points"]
    assert all(NODE_PATH_KEY not in point["metadata"] for point in points)
    assert points[0]["metadata"][PARENT_NODE_ID_KEY] == "article:1"
    assert extract_chunk_context(points[1]["metadata"])[PARENT_NODE_ID_KEY] == "article:2"


def unit(kind: str, number: str) -> dict[str, Any]:
    return {"heading": f"{kind} {number}", "unit_kind": kind, "boundary_value": number, "content": ""}


def node(node_id: str, chars: int) -> DocumentNode:
    return DocumentNode(node_id, None, "point", node_id, None, node_id.count("/") + 1, 0, 0, (), chars, "")


def test_measured_chain_and_chunk_limit_use_current_scope() -> None:
    chain = [node("a/b", 1), node("a", 1)]
    chunks = [("a/b", "xx"), ("a/b/c", "yyy"), ("a", "z"), ("a/bb", "q"), (None, "manual")]

    measured = measured_chain(chain, chunks)

    assert [item.char_count for item in measured] == [5, 7]
    wide = DocumentNode("a", None, "chapter", "a", None, 1, 0, 70, (), 1, "")
    assert within_chunk_limit([node("a/b", 1), wide], 64) == [node("a/b", 1)]
