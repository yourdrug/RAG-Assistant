"""PDF diagnostic boundaries and the rolling summary infrastructure contract."""

from pathlib import Path
from unittest.mock import Mock

import fitz
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import pytest

from domain.knowledge.enums import PageContentType
from infrastructure.ml.extraction.summary_adapter import RollingSummaryUpdater
from infrastructure.ml.guardrails import pdf_adapter


def test_pdf_adapter_reads_renders_and_closes_document(tmp_path: Path) -> None:
    path = tmp_path / "diagnostic.pdf"
    with fitz.Document() as source:
        source.new_page().insert_text((72, 72), "Meaningful document text.")
        source.save(path)

    adapter = pdf_adapter.FitzPDFDocument()
    document = adapter.open(str(path))
    try:
        assert adapter.get_page_count(document) == 1
        assert "Meaningful document text." in adapter.get_page_text(document, 0)
        assert adapter.render_page_image(document, 0, dpi=72).startswith(b"\x89PNG")
        assert not adapter.find_tables(document.load_page(0))
    finally:
        adapter.close(document)

    with pytest.raises(ValueError, match="closed"):
        adapter.get_page_count(document)


def test_table_detection_failure_is_logged_and_classification_remains_available(
    caplog: pytest.LogCaptureFixture,
) -> None:
    page = Mock(find_tables=Mock(side_effect=RuntimeError("table detection unavailable")))
    with caplog.at_level("DEBUG", logger=pdf_adapter.__name__):
        result, _ = pdf_adapter.MLPageClassifier().classify_page("", 0, page)
    assert result == PageContentType.EMPTY.value
    assert "PDF table detection failed" in caplog.text
    page.find_tables.assert_called_once()


def test_ocr_adapter_preserves_pdf_pages_and_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    document = Mock()
    ocr = Mock(return_value={2: "recognized page"})
    monkeypatch.setattr(pdf_adapter, "ocr_pdf_pages", ocr)
    assert pdf_adapter.MLOcrRunner().ocr_pages(document, [2], "scan.pdf") == {2: "recognized page"}
    ocr.assert_called_once_with(document, [2], "scan.pdf")


@pytest.mark.asyncio
async def test_summary_adapter_composes_registered_runnable() -> None:
    prompts: list[str] = []

    def respond(request: object) -> AIMessage:
        prompts.append(str(request))
        return AIMessage(content="Updated summary")

    clients = Mock(llm=Mock(return_value=RunnableLambda(respond)))
    result = await RollingSummaryUpdater(clients).update(
        "Earlier summary", [{"role": "user", "content": "New question"}]
    )
    assert result == "Updated summary"
    assert len(prompts) == 1
    assert "Earlier summary" in prompts[0]
    assert "New question" in prompts[0]


@pytest.mark.asyncio
async def test_summary_adapter_propagates_empty_response_failure() -> None:
    clients = Mock(llm=Mock(return_value=RunnableLambda(lambda _: AIMessage(content=" "))))
    with pytest.raises(ValueError, match="empty or non-text"):
        await RollingSummaryUpdater(clients).update(None, [{"role": "user", "content": "question"}])
