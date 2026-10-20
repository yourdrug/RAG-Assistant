"""Unsupported domain values must never be attached to ingestion chunks."""

from types import SimpleNamespace

import pytest

from application.ingestion import S3DocumentLoader
from domain.ingestion.raw_document import RawDocument
from domain.knowledge.enums import DocDomain


class PassthroughSplitter:
    def split(self, docs, context):
        return docs


@pytest.fixture
def loader():
    return S3DocumentLoader(None, None, PassthroughSplitter(), SimpleNamespace())


def test_unknown_explicit_domain_is_rejected_by_loader(loader):
    with pytest.raises(ValueError):
        loader.classify_text_domain("document text", "legal_typo")


def test_direct_indexing_cannot_save_unknown_domain_metadata(loader):
    with pytest.raises(ValueError):
        loader.index_docs([RawDocument("document text", {})], domain="legal_typo")


@pytest.mark.parametrize("domain", list(DocDomain))
def test_known_domain_is_preserved_in_chunks(loader, domain):
    resolved = loader.classify_text_domain("document text", domain.value)
    chunks = loader.index_docs([RawDocument("document text", {})], domain=resolved)
    assert chunks[0].metadata["doc_domain"] == domain.value
