"""Regressions for PDF instruction packing, document identity and unmatched boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import fitz
from langchain.schema import Document
import pytest

from domain.ingestion.content_splitter import split_by_content
from domain.ingestion.profiles.contract import ContractDomainProfile
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from infrastructure.ml.ingestion.splitting import Chunker, merge_inferred_sections, split_documents
from infrastructure.ml.langchain_document_parser import LangchainDocumentParser, LangchainDocumentSplitter


if TYPE_CHECKING:
    from pathlib import Path


def pdf_instruction(path: Path, title: str = "") -> None:
    with fitz.Document() as pdf:
        for heading, body in [
            ("1. Create document", "Select the document type."),
            ("2. Send document", "Sign and send the document."),
        ]:
            page = pdf.new_page()
            page.insert_text((72, 80), heading, fontsize=18)
            page.insert_text((72, 120), body, fontsize=11)

        if title:
            pdf.set_metadata({"title": title})

        pdf.save(path)


@pytest.mark.parametrize("title", ["", "Document creation guide"])
def test_short_pdf_guide_keeps_title_steps_and_pages(tmp_path: Path, title: str) -> None:
    path = tmp_path / "guide.pdf"
    pdf_instruction(path, title)
    raw = LangchainDocumentParser().parse(path, filename="original.pdf")
    assert {doc.metadata["doc_title"] for doc in raw} == {title or "original"}
    chunks = LangchainDocumentSplitter().split(raw)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.metadata["pages"] == [1, 2]
    assert chunk.metadata["step_numbers"] == ["1", "2"]
    assert "1. Create document" in chunk.page_content
    assert "2. Send document" in chunk.page_content
    assert (chunk.embed_text or "").startswith(title or "original")


def test_unnumbered_pdf_section_is_not_document_title(tmp_path: Path) -> None:
    path = tmp_path / "fees.pdf"

    with fitz.Document() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 80), "SERVICE FEES", fontsize=18)
        page.insert_text((72, 120), "The registration fee applies.", fontsize=11)
        page.insert_text((72, 180), "EXEMPTIONS", fontsize=18)
        page.insert_text((72, 220), "Existing customers are exempt.", fontsize=11)
        pdf.save(path)

    raw = LangchainDocumentParser().parse(path)
    assert {doc.metadata["doc_title"] for doc in raw} == {"fees"}


@pytest.mark.parametrize("barrier", ["maximum", "table", "explicit", "source"])
def test_pdf_guide_packing_respects_limits_and_boundaries(barrier: str) -> None:
    docs = [
        Document(
            page_content=f"{number}. Step\n" + ("Body. " * 100 if barrier == "maximum" else "Body."),
            metadata={
                "source": f"guide-{number}.pdf" if barrier == "source" else "guide.pdf",
                "page": number,
                "section": f"{number}. Step",
                "heading": f"{number}. Step",
                "inferred_heading": barrier != "explicit",
            },
        )
        for number in (1, 2)
    ]

    if barrier == "table":
        docs.insert(
            1,
            Document(
                page_content="| Fee |\n| --- |\n| 50 |",
                metadata={
                    "source": "guide.pdf",
                    "page": 1,
                    "inferred_heading": True,
                    "content_type": "table",
                },
            ),
        )

    chunker = Chunker("general", token_counter=SimpleNamespace(count=len))
    assert merge_inferred_sections(docs, chunker) == docs


@pytest.mark.parametrize("profile", [LegalDomainProfile(), DecreeDomainProfile(), ContractDomainProfile()])
def test_unmatched_structural_boundaries_keep_preamble_type(
    profile: LegalDomainProfile | DecreeDomainProfile | ContractDomainProfile,
) -> None:
    text = "  This introduction has no structural markers.  "
    units = split_by_content(text, profile.content_boundaries(), 1000)
    assert len(units) == 1
    unit = units[0]
    assert unit.unit_kind == "preamble"
    assert unit.heading is None
    assert unit.boundary_value is None
    assert text[unit.start : unit.end] == unit.content
    chunks = split_documents([Document(page_content=text)], profile=profile)
    assert chunks[0].metadata["unit_kind"] == "preamble"
