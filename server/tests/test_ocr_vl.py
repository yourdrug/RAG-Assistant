"""Local OCR contract, scan routing, preview parity and bounded table retries."""

# ruff: noqa: RUF001 -- Cyrillic OCR fixtures intentionally contain Russian words.

from collections.abc import Callable
import io
from pathlib import Path
import sys
from unittest.mock import Mock

import fitz  # type: ignore[import-untyped]  # PyMuPDF does not ship typing metadata.
import httpx
import numpy as np
from PIL import Image
import pytest

from config import settings
from domain.ingestion.preview import DryRunPageResult
from domain.knowledge.enums import PageContentType
from infrastructure.ml.ingestion import ocr
from infrastructure.ml.ingestion.ocr_vl import OCRPage, OCRServiceError, recognize_page
from infrastructure.ml.ingestion.pdf import parse_pdf
from infrastructure.ml.ingestion.splitting import split_documents
from infrastructure.ml.preview import docx_preview_strategy as docx
from infrastructure.ml.preview.pdf_preview_strategy import PdfPreviewStrategy


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "ocr_vl"))
# The standalone service is loaded from the runtime path above, outside the server package.
import service as vl_service  # type: ignore[import-not-found]
from tables import UNREADABLE, table_markdown  # type: ignore[import-not-found]


TABLE = (
    "| Название | Количество | Масса | Проба | Итог |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| Кольцо | 2 | 3,14 | 585 | 6,28 |"
)


@pytest.fixture
def vl_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_engine", "paddleocr_vl")
    monkeypatch.setattr(settings, "ocr_enabled", True)
    monkeypatch.setattr(settings, "ocr_dpi", 72)


def scan_pdf(path: Path) -> Path:
    image = Image.new("RGB", (100, 140), "white")
    data = io.BytesIO()
    image.save(data, "PNG")
    with fitz.open() as document:
        page = document.new_page()
        page.insert_image(page.rect, stream=data.getvalue())
        # A substantial existing text layer must not mask an unread table.
        page.insert_text((20, 20), "Recognized header " * 8)
        document.new_page().insert_text((20, 20), "Native text page " * 8)
        document.save(path)
    return path


def test_preview_and_index_use_same_blocks_and_page(
    vl_engine: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = scan_pdf(tmp_path / "scan.pdf")
    result = OCRPage.model_validate(
        {
            "blocks": [
                {"kind": "text", "text": "Опись"},
                {"kind": "table", "text": TABLE},
            ]
        }
    )
    recognize = Mock(return_value=result)
    monkeypatch.setattr(ocr, "recognize_page", recognize)
    documents = parse_pdf(path)
    assert [doc.metadata["page"] for doc in documents] == [1, 1, 2]
    assert documents[1].metadata["content_type"] == PageContentType.TABLE
    assert documents[1].metadata["source"] == str(path)
    assert documents[1].page_content == TABLE
    assert "Native text page" in documents[2].page_content
    recognize.assert_called_once()
    diagnostic = Mock()
    diagnostic.analyze_text_layer.return_value = (
        [
            DryRunPageResult(page=1, type=PageContentType.TEXT),
            DryRunPageResult(page=2, type=PageContentType.TEXT),
        ],
        {},
        0,
    )
    units, counts, total = PdfPreviewStrategy(diagnostic).analyze(path)
    assert units[0].full_text == "\n\n".join(doc.page_content for doc in documents[:2])
    assert units[0].type == PageContentType.TABLE
    assert total == sum(unit.chars for unit in units)
    assert sum(counts.values()) == 2
    # ACL stamping occurs downstream; the table splitter must retain it.
    documents[1].metadata.update(owner_id=42, visibility="private", group_id=7)
    chunks = split_documents([documents[1]])
    assert chunks

    for chunk in chunks:
        assert "| Кольцо | 2 | 3,14 | 585 | 6,28 |" in chunk.page_content
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["owner_id"] == 42
        assert chunk.metadata["visibility"] == "private"
        assert chunk.metadata["group_id"] == 7
        assert "Название" in chunk.metadata["table_header"]


@pytest.mark.parametrize("failure", ["timeout", "offline", "invalid", "empty", "blank", "http"])
def test_client_errors_are_explicit_and_close_client(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    factory = Mock(return_value=client)
    monkeypatch.setattr(httpx, "Client", factory)

    if failure in ("timeout", "offline"):
        error = httpx.ReadTimeout if failure == "timeout" else httpx.ConnectError
        client.post.side_effect = error("unavailable")
    else:
        response = httpx.Response(
            503 if failure == "http" else 200,
            request=httpx.Request("POST", "http://local"),
            json={"blocks": []}
            if failure == "empty"
            else {"blocks": [{"kind": "text", "text": " "}]}
            if failure == "blank"
            else {"oops": True},
        )
        client.post.return_value = response

    with pytest.raises(OCRServiceError, match="Локальный OCR"):
        recognize_page(Image.new("RGB", (10, 10)))
    client.__exit__.assert_called_once()
    assert factory.call_args.kwargs["trust_env"] is False


def test_client_sends_pixels_and_accepts_structured_result(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://local"),
        json={"blocks": [{"kind": "table", "text": TABLE}]},
    )
    client.post.return_value = response
    monkeypatch.setattr(httpx, "Client", Mock(return_value=client))
    assert recognize_page(Image.new("RGB", (10, 10))).blocks[0].text == TABLE
    assert client.post.call_args.kwargs["content"].startswith(b"\x89PNG")


def test_service_retries_incomplete_table_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = Mock()
    first = {"block_label": "table", "block_content": "", "block_bbox": [0, 0, 10, 10]}
    retry = {
        "block_label": "table",
        "block_content": "<table><tr><td>Масса</td></tr><tr><td>�</td></tr></table>",
    }
    predict = Mock(side_effect=[([first], {}), ([retry], {})])
    monkeypatch.setattr(vl_service, "predict_blocks", predict)
    result = vl_service.recognize(pipeline, np.zeros((20, 20, 3), dtype=np.uint8))
    assert predict.call_count == 2
    assert predict.call_args.kwargs == {"preprocess": False}
    assert UNREADABLE in result["blocks"][0]["text"]


def test_service_preserves_reading_order_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = [
        {"block_label": "text", "block_content": "Опись"},
        {"block_label": "table", "block_content": TABLE},
    ]
    predict = Mock(return_value=(blocks, {}))
    monkeypatch.setattr(vl_service, "predict_blocks", predict)
    result = vl_service.recognize(Mock(), np.zeros((20, 20, 3), dtype=np.uint8))
    assert [block["kind"] for block in result["blocks"]] == ["text", "table"]
    predict.assert_called_once()


def test_html_table_merged_cells_keep_column_alignment() -> None:
    content = (
        '<table><tr><th rowspan="2">Название</th><th colspan="2">Масса</th></tr>'
        "<tr><th>До</th><th>После</th></tr>"
        "<tr><td>Кольцо</td><td>3,14</td><td>2,99</td></tr></table>"
    )
    result = table_markdown(content)
    assert "| Название | Масса | Масса |" in result
    assert "| Название | До | После |" in result
    assert "| Кольцо | 3,14 | 2,99 |" in result


@pytest.mark.asyncio
async def test_service_rejects_non_image_and_reports_model_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    vl_service.app.state.pipeline = Mock()
    transport = httpx.ASGITransport(app=vl_service.app)

    async def run_inline(function: Callable[..., object], *args: object) -> object:
        return function(*args)

    monkeypatch.setattr(vl_service, "run_in_threadpool", run_inline)
    async with httpx.AsyncClient(transport=transport, base_url="http://local") as client:
        response = await client.post("/recognize", content=b"bad", headers={"Content-Type": "image/png"})
        assert response.status_code == 422
        response = await client.post("/recognize", json={"url": "https://example.com"})
        assert response.status_code == 415

        async def fail_inference(*args: object) -> None:
            message = "private content"
            raise RuntimeError(message)

        monkeypatch.setattr(vl_service, "run_in_threadpool", fail_inference)
        data = io.BytesIO()
        Image.new("RGB", (10, 10)).save(data, "PNG")
        response = await client.post(
            "/recognize", content=data.getvalue(), headers={"Content-Type": "image/png"}
        )
        assert response.status_code == 503
        assert "private content" not in response.text


def test_model_result_contract_and_preprocessing() -> None:
    pipeline = Mock()
    prediction = Mock()
    prediction.json = {
        "res": {
            "parsing_res_list": [
                {"block_label": "table", "block_content": TABLE},
            ]
        }
    }
    pipeline.predict.return_value = [prediction]
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    blocks, result = vl_service.predict_blocks(pipeline, image)
    assert result is prediction
    assert blocks[0]["block_content"] == TABLE
    assert pipeline.predict.call_args.kwargs == {
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": True,
        "format_block_content": True,
    }


def test_service_releases_lock_on_error_and_rejects_concurrent_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setattr(vl_service, "predict_blocks", Mock(side_effect=ValueError))
    with pytest.raises(ValueError, match=r"^$"):
        vl_service.recognize(Mock(), image)
    assert vl_service.inference_lock.acquire(blocking=False)
    try:
        with pytest.raises(vl_service.HTTPException) as error:
            vl_service.recognize(Mock(), image)
        assert error.value.status_code == 503
    finally:
        vl_service.inference_lock.release()


@pytest.mark.parametrize("content", ["| A | B |\n| 1 | 2 |", "| A | B |\n| --- | --- |\n| 1 |"])
def test_malformed_markdown_tables_require_retry(content: str) -> None:
    assert table_markdown(content) == ""


@pytest.mark.parametrize("failed", [False, True])
def test_docx_uses_selected_local_engine(
    vl_engine: None, monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    data = io.BytesIO()
    Image.new("RGB", (10, 10)).save(data, "PNG")
    monkeypatch.setattr(docx, "_extract_images_without_alt", Mock(return_value=[data.getvalue()]))
    runner = (
        Mock(side_effect=OCRServiceError)
        if failed
        else Mock(
            return_value=OCRPage.model_validate(
                {
                    "blocks": [{"kind": "table", "text": TABLE}],
                }
            )
        )
    )
    monkeypatch.setattr(docx, "recognize_page", runner)
    units = [DryRunPageResult(page=1, type=PageContentType.IMAGE_ONLY)]

    if failed:
        with pytest.raises(OCRServiceError):
            docx.DocxPreviewStrategy().ocr_problem_units(Path("file.docx"), units, [1])
    else:
        result, _, _ = docx.DocxPreviewStrategy().ocr_problem_units(Path("file.docx"), units, [1])
        assert result[0].full_text == TABLE

    runner.assert_called_once()
