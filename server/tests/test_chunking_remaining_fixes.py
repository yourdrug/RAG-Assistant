"""Regression scenarios for mixed PDF pages, instruction scope and prompt budgets."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from docx import Document as WordDocument
import fitz
from langchain.schema import Document
import pytest

from application.rag.stream_events import StatusEvent
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.registry import DomainProfileRegistry
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.enums import DocDomain, PageContentType
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.exceptions import ContextBudgetExceededError
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.ingestion.pdf import parse_pdf, pymupdf_table_to_markdown
from infrastructure.ml.ingestion.splitting import split_documents
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter
from infrastructure.ml.rag.rag_formatting import CHARS_PER_TOKEN, format_docs, format_docs_with_selection
from presentation.api.routes.chat import chat_stream
from presentation.api.schemas import ChatRequest, CurrentUser
from tests.rag_stage_fixtures import build_context as step_build_context, stage_fixture
from tests.test_rag_pipeline import make_rag


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from application.rag.stream_events import StreamEvent


@pytest.fixture
def pipeline() -> tuple[LangchainDocumentParser, LangchainDocumentSplitter]:
    settings = DomainSettingsAdapter()
    values = {}
    registry = DomainProfileRegistry()
    profile = DecreeDomainProfile(settings)
    registry.register(profile)

    for default in profile.config_defaults():
        values[default.key, profile.key] = default.value

    settings.replace_snapshot(values)
    return LangchainDocumentParser(registry, settings), LangchainDocumentSplitter(registry, settings)


def test_mixed_pdf_keeps_text_before_between_and_after_tables(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdf"
    with fitz.Document() as pdf:
        page = pdf.new_page()
        page.insert_text((60, 90), "Before: fees apply only to new applications.", fontsize=10)
        page.insert_text((300, 115), "Before-right: additional eligibility.", fontsize=10)

        for top in (150, 300):
            for y in (top, top + 30, top + 60):
                page.draw_line((60, y), (440, y))

            for x in (60, 250, 440):
                page.draw_line((x, top), (x, top + 60))

            for x, y, value in (
                (70, top + 20, "Service"),
                (260, top + 20, "Fee"),
                (70, top + 50, f"Item-{top}"),
                (260, top + 50, "50"),
            ):
                page.insert_text((x, y), value, fontsize=10)

        page.insert_text((60, 260), "Between: the second table covers additional services.", fontsize=10)
        page.insert_text((60, 400), "After: existing customers are exempt from payment.", fontsize=10)
        assert len(page.find_tables().tables) == 2
        pdf.save(str(path))
    docs = parse_pdf(path)
    chunks = split_documents(docs)
    content = "\n".join(c.page_content for c in chunks)
    assert content.index("Before:") < content.index("Item-150") < content.index("Between:")
    assert content.index("Before-right:") < content.index("Item-150")
    assert content.index("Between:") < content.index("Item-300") < content.index("After:")
    assert content.count("Item-150") == content.count("Item-300") == 1
    assert all(c.metadata["page"] == 1 for c in chunks)
    assert all(not any(key.startswith("_table") for key in c.metadata) for c in chunks)


def test_multiline_docx_cells_remain_one_logical_row(
    pipeline: tuple[LangchainDocumentParser, LangchainDocumentSplitter], tmp_path: Path
) -> None:
    parser, splitter = pipeline
    word = WordDocument()
    table = word.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Service\ncategory"
    table.cell(0, 1).text = "Fee"
    table.cell(1, 0).text = "Registration"
    table.cell(1, 1).text = "50\nTax included"
    path = tmp_path / "multiline.docx"
    word.save(str(path))
    chunks = splitter.split(parser.parse(path))
    assert len(chunks) == 1
    assert chunks[0].metadata["table_row_start"] == chunks[0].metadata["table_row_end"] == 1
    lines = chunks[0].page_content.splitlines()
    assert len(lines) == 3
    assert "Service" in lines[0]
    assert "category" in lines[0]
    assert "Registration" in lines[2]
    assert "50" in lines[2]
    assert "Tax included" in lines[2]
    assert chunks[0].metadata["table_header"] == "\n".join(lines[:2])


def test_multiline_pdf_cells_remain_one_logical_row() -> None:
    table = SimpleNamespace(extract=lambda: [["Service\ncategory", "Fee"], ["A", "50\nTax included"]])
    chunks = split_documents(
        [
            Document(
                page_content=pymupdf_table_to_markdown(table),
                metadata={"content_type": PageContentType.TABLE.value},
            )
        ]
    )
    assert len(chunks) == 1
    assert chunks[0].metadata["table_row_end"] == 1
    assert len(chunks[0].page_content.splitlines()) == 3


def test_each_instruction_step_retains_the_complete_intro(
    pipeline: tuple[LangchainDocumentParser, LangchainDocumentSplitter],
) -> None:
    _, splitter = pipeline
    condition = "Only existing customers may use this procedure."
    intro = condition + "\n" + "Review the eligibility conditions before continuing. " * 8
    text = intro + "\n1. Send application\n" + "Fill in all required fields. " * 11
    text += "\n2. Pay fee\n" + "Pay the fee using a bank transfer. " * 9
    chunks = splitter.split([RawDocument(text, {"source": "guide.txt"})])
    steps = [c for c in chunks if c.metadata.get("step_number")]
    assert steps
    assert all(
        condition
        in format_docs([Document(page_content=item.page_content, metadata=item.metadata) for item in [c]])
        for c in steps
    )
    assert all(c.metadata["parent_units"][-1]["content"] == intro.strip() for c in steps)
    assert all(len(c.page_content) <= 550 for c in chunks)


def test_packed_steps_keep_intro_even_when_it_is_in_a_separate_chunk(
    pipeline: tuple[LangchainDocumentParser, LangchainDocumentSplitter],
) -> None:
    _, splitter = pipeline
    intro = "Use these steps only after approval. " * 16
    text = intro + "\n1. Open\nOpen the card.\n2. Send\nSend the application."
    chunks = splitter.split([RawDocument(text, {"source": "guide.txt"})])
    packed = next(c for c in chunks if c.metadata.get("step_numbers") == ["1", "2"])
    assert intro.strip() in format_docs(
        [Document(page_content=item.page_content, metadata=item.metadata) for item in [packed]]
    )


def test_oversized_parent_is_never_truncated_or_allowed_to_exceed_budget(
    pipeline: tuple[LangchainDocumentParser, LangchainDocumentSplitter],
) -> None:
    _, splitter = pipeline
    intro = "21. " + "Eligibility conditions apply to applications. " * 700
    text = intro + "\n21.1. Submit the passport.\n21.2. Submit the form."
    child = next(
        c
        for c in splitter.split([RawDocument(text, {"source": "rules.txt"})], DocDomain.DECREE.value)
        if c.metadata.get("subpoint_num_number") == "21.2"
    )
    context, selected = format_docs_with_selection(
        [Document(page_content=item.page_content, metadata=item.metadata) for item in [child]],
        max_context_tokens=6000,
    )
    assert len(context) <= 6000 * CHARS_PER_TOKEN
    assert selected == []
    assert context == ""


def test_oversized_document_does_not_discard_later_usable_sources() -> None:
    oversized = Document(page_content="large " * 100, metadata={"source": "large.txt"})
    small = Document(page_content="Complete statement.", metadata={"source": "small.txt"})
    context, selected = format_docs_with_selection([(oversized, 0.9), (small, 0.8)], max_context_tokens=30)
    assert len(context) <= 30 * CHARS_PER_TOKEN
    assert [(doc.page_content, score) for doc, score in selected] == [(small.page_content, 0.8)]
    assert selected[0][0].metadata["citation_id"] == 1
    assert "[1] small.txt" in context
    assert "[2]" not in context


def test_budget_includes_separators() -> None:
    docs = [Document(page_content=letter * 14, metadata={"source": "a"}) for letter in ("x", "y")]
    context, selected = format_docs_with_selection(
        [Document(page_content=item.page_content, metadata=item.metadata) for item in docs],
        max_context_tokens=10,
    )
    assert len(context) <= 10 * CHARS_PER_TOKEN
    assert len(selected) == 1


@pytest.mark.parametrize("tokens", [0, -1])
def test_nonpositive_context_budget_is_rejected(tokens: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        format_docs_with_selection([], max_context_tokens=tokens)


def test_no_usable_context_stops_before_generation() -> None:

    state = stage_fixture(
        rag=make_rag(),
        t_pipeline_start=0,
        question="Explain the eligibility rules.",
        ctx=RagContext(
            user=UserContext(user_id=12, user_kind=UserKind.INTERNAL, user_role=UserRole.USER, group_ids=())
        ),
        user=UserContext(12, UserKind.INTERNAL, UserRole.USER),
        access_filter=None,
        retrieval_filter=None,
        breadth=Breadth.NARROW,
        docs=[(Document(page_content="Complete conditions. " * 1500, metadata={"source": "rules.pdf"}), 0.9)],
    )
    with pytest.raises(ContextBudgetExceededError):
        step_build_context(state, None)
    assert state._prompt_docs == []


@pytest.mark.asyncio
async def test_stream_reports_context_budget_error_without_answer_or_sources() -> None:

    async def stream_chat(*args: object, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        yield StatusEvent(stage="searching")
        raise ContextBudgetExceededError

    response = await chat_stream(
        ChatRequest(question="Укажите порядок подачи заявления."),
        MagicMock(is_disconnected=AsyncMock(return_value=False)),
        current_user=CurrentUser(
            id=12, email="user@example.test", is_active=True, kind=UserKind.INTERNAL, role=UserRole.USER
        ),
        chat_service=MagicMock(stream_chat=stream_chat),
        log=MagicMock(),
    )
    events = "".join([event async for event in response.body_iterator])
    assert "event: error" in events
    assert ContextBudgetExceededError.code in events
    assert "Уточните вопрос" in events
    assert '"text"' not in events
    assert "event: done" not in events
