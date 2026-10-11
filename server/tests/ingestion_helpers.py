"""Storage and ingestion collaborators for source-identity contract tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from application.ports.file_storage import FileItem
from application.services.batch_ingestion import BatchIngestionWorkflow
from application.services.document_loader import S3DocumentLoader
from application.services.ingestion_orchestrator import IngestionService
from application.services.ingestion_registry import IngestionRegistry
from application.services.ingestion_sync import DocumentSyncService
from application.services.ingestion_targets import S3IngestionTargets, S3UploadService
from application.services.single_file_ingestion import SingleFileIngestionWorkflow
from application.services.sparse_index_builder import SparseIndexBuilder
from domain.entities.raw_document import RawDocument
from fakes import FakeUnitOfWorkFactory

FIRST_KEY = "docs/a/manual.md"
SECOND_KEY = "docs/b/manual.md"
FIRST_SOURCE = "s3://bucket/docs/a/manual.md"
SECOND_SOURCE = "s3://bucket/docs/b/manual.md"
FILENAME = "manual.md"


class MemoryStorage:
    """Exercise storage-key decisions while avoiding external S3."""

    supported_extensions = (".md",)

    def __init__(self, tmp_path):
        self._temp_dir = tmp_path
        self._downloads = 0
        self.contents = {FIRST_KEY: "first text", SECOND_KEY: "other text"}
        self.revisions = dict.fromkeys(self.contents, "same-mtime")

    def get_file_info(self, key):
        if key not in self.contents:
            return None
        return FileItem(key, key.rsplit("/", 1)[-1], len(self.contents[key]), self.revisions[key], ".md")

    def list_files(self, prefix):
        return [self.get_file_info(key) for key in self.contents if key.startswith(prefix)]

    async def download_to_temp(self, key):
        self._downloads += 1
        path = self._temp_dir / f"download-{self._downloads}.md"
        path.write_text(self.contents[key])
        return path


class TextParser:
    def parse(self, path, meta):
        return [RawDocument(path.read_text(), {"source": meta.source_key, "filename": meta.filename})]


class IdentitySplitter:
    def split(self, documents, context):
        return documents


class SparseProjection:
    def __init__(self):
        self.texts = []

    async def rebuild(self, chunks):
        self.texts = [chunk.text for chunk in chunks]

    async def extend(self, chunks):
        self.texts.extend(chunk.text for chunk in chunks)


def ingestion_components(tmp_path):
    factory = FakeUnitOfWorkFactory()
    storage = MemoryStorage(tmp_path)
    settings = SimpleNamespace(
        s3_bucket="bucket",
        hybrid_enabled=True,
        embed_dim=3,
        tei_embed_url="",
        qdrant_url="",
        collection_name="test",
    )
    registry = IngestionRegistry(factory, storage)
    loader = S3DocumentLoader(storage, TextParser(), IdentitySplitter(), settings, registry=registry)
    sync = DocumentSyncService(factory)
    sparse = SparseProjection()
    vectors = MagicMock(ensure_collection=AsyncMock())
    batch = BatchIngestionWorkflow(
        vectors, settings, loader, registry, sync, SparseIndexBuilder(settings, sparse)
    )
    single = SingleFileIngestionWorkflow(storage, loader, registry, sync)
    targets = S3IngestionTargets()
    service = IngestionService(batch, single, registry, targets, S3UploadService(storage, targets))
    return service, factory, storage, sparse


def sync_entries(keys=(FIRST_KEY, SECOND_KEY)):
    return {key: {"source": f"s3://bucket/{key}", "filename": key.rsplit("/", 1)[-1]} for key in keys}


def source_chunks(keys=(FIRST_KEY, SECOND_KEY)):
    return [
        RawDocument(text, {"source": f"s3://bucket/{key}"})
        for key, text in zip(keys, ("first text", "other text"), strict=False)
    ]
