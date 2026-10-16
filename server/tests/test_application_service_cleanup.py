"""Preserve metrics and PDF recommendations while removing lint exceptions."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from application.ports.metrics_registry import MetricsRegistryPort
from application.services.metrics_service import MetricsService
from application.services.pdf_diagnostic_service import PDFDiagnosticService
from domain.value_objects.page_content_type import PageContentType
from domain.value_objects.pdf_dto import DryRunPageResult


@pytest.mark.parametrize(
    "total, bad_pages, page_type, expected",
    [
        (0, [], PageContentType.SCAN, "Документ пуст"),
        (10, [1], PageContentType.SCAN, "Качество хорошее"),
        (10, [1, 2], PageContentType.GARBLED, "OCR может не помочь"),
        (10, [9, 10], PageContentType.SCAN, "начиная от страницы 9"),
        (20, [1, 3, 5], PageContentType.SCAN, "точечный OCR"),
        (10, [1, 3, 5], PageContentType.SCAN, "полную переконвертацию"),
    ],
)
def test_pdf_recommendations(
    total: int, bad_pages: list[int], page_type: PageContentType, expected: str
) -> None:
    pages = [
        DryRunPageResult(
            page=number,
            type=page_type if number in bad_pages else PageContentType.TEXT,
            content_type=PageContentType.TEXT,
            chars=0,
            preview="",
            full_text="",
        )
        for number in range(1, total + 1)
    ]
    counts = {page_type.value: len(bad_pages)}

    assert expected in PDFDiagnosticService.suggest_action(pages, counts)


def test_metrics_preserve_totals_histograms_and_status_keys() -> None:
    registry = Mock(spec=MetricsRegistryPort)
    registry.collect_gauge.return_value = {}
    counters = {
        "rag_queries": {"model_one": 2.0, "model_two": 3.0},
        "rag_not_found": {"model_one": 1.0},
        "ingest_documents": {"worker_success": 4.0, "failed": 2.0, "worker_": 1.0},
        "http_requests": {"GET_health": 7.0},
    }
    registry.collect_counter.side_effect = lambda name: counters.get(name, {})
    registry.collect_histogram.side_effect = lambda name: {"metric": name}

    result = MetricsService(registry).collect()

    assert result.rag == {
        "queries_total": 5.0,
        "not_found_total": 1.0,
        "stage_latency": {"metric": "rag_stage_duration_seconds"},
        "answer_length": {"metric": "rag_answer_length_chars"},
        "retrieved_chunks": {"metric": "rag_retrieved_chunks"},
    }
    assert result.ingestion["by_status"] == {"success": 4.0, "failed": 2.0, "": 1.0}
    assert result.http_requests == {"total": 7.0, "by_endpoint": {"GET_health": 7.0}}
    assert result.qdrant == {"points": 0.0}
    assert result.bm25 == {"index_size": 0.0}
