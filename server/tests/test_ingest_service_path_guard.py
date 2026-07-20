"""
Тесты для IngestService._resolve_within_data_dir / resolve_ingest_target / resolve_docs_dir —
защиты POST /ingest и POST /ingest/file от path traversal. Не трогают Qdrant/эмбеддинги/
S3 — только разрешение путей на локальном FILE_BACKEND.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest  # noqa: E402
from config import settings  # noqa: E402
from services.ingest_service import IngestService  # noqa: E402


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings, "file_backend", "local")
    (tmp_path / "docs_sample").mkdir()
    return IngestService()


def test_relative_docs_dir_inside_data_dir_resolves(service, tmp_path):
    resolved = service.resolve_docs_dir("docs_sample")
    assert resolved == str((tmp_path / "docs_sample").resolve())


def test_absolute_docs_dir_inside_data_dir_resolves(service, tmp_path):
    inside = tmp_path / "docs_sample"
    assert service.resolve_docs_dir(str(inside)) == str(inside.resolve())


def test_docs_dir_traversal_outside_data_dir_is_rejected(service):
    with pytest.raises(ValueError):
        service.resolve_docs_dir("../../../etc")


def test_docs_dir_absolute_outside_data_dir_is_rejected(service):
    with pytest.raises(ValueError):
        service.resolve_docs_dir("/etc")


def test_docs_dir_sibling_with_shared_prefix_is_rejected(service, tmp_path):
    """'<data_dir>-evil' не должен проходить только из-за строкового префикса."""
    sibling = tmp_path.parent / (tmp_path.name + "-evil")
    with pytest.raises(ValueError):
        service.resolve_docs_dir(str(sibling))


def test_file_path_traversal_outside_data_dir_is_rejected(service):
    with pytest.raises(ValueError):
        service.resolve_ingest_target("../../../etc/passwd")


def test_file_path_inside_data_dir_resolves_when_it_exists(service, tmp_path):
    target = tmp_path / "docs_sample" / "report.pdf"
    target.write_bytes(b"%PDF-1.4")
    assert service.resolve_ingest_target(str(target)) == str(target.resolve())


def test_file_path_inside_data_dir_but_missing_raises_not_found(service):
    with pytest.raises(FileNotFoundError):
        service.resolve_ingest_target("docs_sample/does-not-exist.pdf")
