"""RAG-002: database identity, concurrent creation and legacy migration."""

import asyncio
import importlib
from contextlib import asynccontextmanager

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from application.services.ingestion_sync import DocumentSyncService
from domain.entities.document import Document
from domain.entities.raw_document import RawDocument
from domain.entities.vector_outbox_entry import OutboxOperation, OutboxStatus
from domain.repositories.ingestion_registry_repository import IngestionRegistryEntry
from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.source_type import SourceType
from domain.value_objects.visibility import DocumentVisibility
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.database.models import DocumentModel, UserModel
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository
from infrastructure.repositories.document.sqlalchemy_ingestion_registry_repository import (
    SQLAlchemyIngestionRegistryRepository,
)
from infrastructure.uow_factory import UnitOfWorkFactory

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
REVISION_MODULE = "infrastructure.database.migrations.versions.n4e5f6g7h8i9_ingestion_object_identity"
FIRST_KEY = "docs/a/manual.md"
SECOND_KEY = "docs/b/manual.md"
FILENAME = "manual.md"


@pytest.mark.parametrize("keys", [(FIRST_KEY, FIRST_KEY), (FIRST_KEY, SECOND_KEY)])
async def test_concurrent_document_creation_preserves_source_identity(pg_engine, keys):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)

    class SessionDatabase:
        def get_session(self, master=False):
            return sessions()

    real_factory = UnitOfWorkFactory(SessionDatabase())
    barrier = asyncio.Barrier(2)
    attempts = 0

    class GatedRepository(SQLAlchemyDocumentRepository):
        async def save(self, document):
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                await barrier.wait()
            return await super().save(document)

    class GatedFactory:
        @asynccontextmanager
        async def create(self, master=False):
            async with real_factory.create(master=master) as uow:
                uow.documents = GatedRepository(uow._session)
                yield uow

    sync = DocumentSyncService(GatedFactory())

    async def create(key):
        return await sync.ensure_document_rows({key: {"source": f"s3://bucket/{key}"}})

    results = await asyncio.wait_for(asyncio.gather(*(create(key) for key in keys)), timeout=10)
    async with sessions() as session:
        rows = (await session.execute(select(DocumentModel))).scalars().all()
        assert len(rows) == len(set(keys))
        assert {row.ingestion_key for row in rows} == set(keys)
        assert len({row.filename for row in rows}) == len(rows)
        assert {next(iter(result.values())).id for result in results} == {row.id for row in rows}


async def test_concurrent_registry_upserts_keep_equal_basenames(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)

    async def upsert(key):
        async with sessions.begin() as session:
            await SQLAlchemyIngestionRegistryRepository(session).upsert(
                IngestionRegistryEntry(key, FILENAME, "hash", f"s3://bucket/{key}", 1, 10)
            )

    await asyncio.gather(upsert(FIRST_KEY), upsert(SECOND_KEY), upsert(FIRST_KEY))
    async with sessions() as session:
        repository = SQLAlchemyIngestionRegistryRepository(session)
        entries = await repository.list_all()
        assert set(entries) == {FIRST_KEY, SECOND_KEY}
        assert {entry.filename for entry in entries.values()} == {FILENAME}
        await repository.delete(FIRST_KEY)
        assert await repository.is_already_indexed(SECOND_KEY, "hash")
        assert not await repository.is_already_indexed(FIRST_KEY, "hash")


async def test_registry_round_trips_maximum_length_object_key(pg_engine):
    key = "docs/" + "a" * 1009 + "/manual.md"
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repository = SQLAlchemyIngestionRegistryRepository(session)
        await repository.upsert(IngestionRegistryEntry(key, FILENAME, "hash", f"s3://bucket/{key}", 1, 10))
        entry = await repository.get(key)
        assert entry.object_key == key
        assert entry.source == f"s3://bucket/{key}"


@pytest.mark.parametrize("reserve_alias", [False, True], ids=["existing-collision", "concurrent-reservation"])
async def test_ownerless_conversion_resolves_filename_atomically(pg_engine, reserve_alias):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        client = UserModel(
            email="identity-client@example.org",
            hashed_password="test",
            role=UserRole.USER.value,
            kind=UserKind.CLIENT.value,
        )
        session.add(client)
        await session.flush()
        client_id = client.id
        repository = SQLAlchemyDocumentRepository(session)
        original = await repository.save(Document(filename=FILENAME, ingestion_key=FIRST_KEY))
        # The intended destination alias is also occupied in the old scope;
        # filename and ACL must change in one SQL update.
        await repository.save(Document(filename="manual(1).md"))
        unrelated = await repository.save(
            Document(filename=FILENAME, owner_id=client_id, visibility=DocumentVisibility.CLIENT_PRIVATE)
        )

    class SessionDatabase:
        def get_session(self, master=False):
            return sessions()

    real_factory = UnitOfWorkFactory(SessionDatabase())
    reserved = False

    class ReservationRepository(SQLAlchemyDocumentRepository):
        async def set_acl(self, document_id, visibility, owner_id, group_id, *, filename=None):
            nonlocal reserved
            if reserve_alias and not reserved and filename == "manual(1).md":
                reserved = True
                async with sessions.begin() as session:
                    await SQLAlchemyDocumentRepository(session).save(
                        Document(
                            filename="manual(1).md",
                            owner_id=client_id,
                            visibility=DocumentVisibility.CLIENT_PRIVATE,
                        )
                    )
            await super().set_acl(document_id, visibility, owner_id, group_id, filename=filename)

    class ReservationFactory:
        @asynccontextmanager
        async def create(self, master=False):
            async with real_factory.create(master=master) as uow:
                uow.documents = ReservationRepository(uow._session)
                yield uow

    source = f"s3://bucket/{FIRST_KEY}"
    await DocumentSyncService(ReservationFactory()).sync_documents_to_db(
        {FIRST_KEY: {"source": source}},
        {},
        [RawDocument("private text", {"source": source})],
        visibility=DocumentVisibility.CLIENT_PRIVATE,
        client_id=client_id,
    )
    async with sessions() as session:
        repository = SQLAlchemyDocumentRepository(session)
        converted = await repository.get_by_id(original.id)
        assert converted.owner_id == client_id
        assert converted.filename == ("manual(2).md" if reserve_alias else "manual(1).md")
        assert converted.ingestion_key == FIRST_KEY
        assert (await repository.get_by_id(unrelated.id)).filename == FILENAME
        assert (
            await session.scalar(
                text("SELECT count(*) FROM chunks WHERE document_id = :id"), {"id": original.id}
            )
            == 1
        )


async def prepare_legacy_schema(pg_engine):
    async with pg_engine.begin() as connection:
        await connection.execute(text("DROP INDEX ux_documents_ingestion_key"))
        await connection.execute(text("ALTER TABLE documents DROP COLUMN ingestion_key"))
        await connection.execute(text("DROP INDEX ux_ingestion_registry_object_key"))
        await connection.execute(text("ALTER TABLE ingestion_registry DROP COLUMN object_key"))
        await connection.execute(
            text("ALTER TABLE ingestion_registry ALTER COLUMN source TYPE varchar(1000)")
        )
        await connection.execute(
            text("CREATE UNIQUE INDEX idx_ingestion_registry_filename ON ingestion_registry (filename)")
        )


async def add_legacy_rows(
    pg_engine, *, source, ambiguous=False, missing_document=False, committed_source=None, source_evidence=True
):
    async with pg_engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO ingestion_registry (filename, source, file_hash, creation_date) "
                "VALUES (:filename, :source, 'hash', NOW())"
            ),
            {"filename": FILENAME, "source": source},
        )
        if not missing_document:
            await connection.execute(
                text(
                    "INSERT INTO documents "
                    "(filename, source_path, visibility, status, doc_domain, source_type, creation_date) "
                    "VALUES (:filename, '', :visibility, :status, :domain, :source_type, NOW())"
                ),
                {
                    "filename": FILENAME,
                    "visibility": DocumentVisibility.INTERNAL_PUBLIC.value,
                    "status": DocumentStatus.DONE.value,
                    "domain": DocDomain.GENERAL.value,
                    "source_type": SourceType.FILE.value,
                },
            )
        if ambiguous:
            # INDEXING is outside the historical filename uniqueness predicate,
            # so two candidate rows are possible and must not be guessed between.
            await connection.execute(
                text(
                    "INSERT INTO documents "
                    "(filename, source_path, visibility, status, doc_domain, source_type, creation_date) "
                    "VALUES (:filename, '', :visibility, :status, :domain, :source_type, NOW())"
                ),
                {
                    "filename": FILENAME,
                    "visibility": DocumentVisibility.INTERNAL_PUBLIC.value,
                    "status": DocumentStatus.INDEXING.value,
                    "domain": DocDomain.GENERAL.value,
                    "source_type": SourceType.FILE.value,
                },
            )
        if source_evidence and not missing_document:
            ids = (await connection.execute(text("SELECT id FROM documents"))).scalars().all()
            for doc_id in ids:
                await connection.execute(
                    text(
                        "INSERT INTO vector_store_outbox "
                        "(operation, aggregate_type, aggregate_id, payload, creation_date, "
                        "status, attempts, max_attempts, next_attempt_at) "
                        "VALUES (:operation, 'document', :doc_id, "
                        "jsonb_build_object('source', CAST(:source AS text), "
                        "'document_id', CAST(:doc_id AS int)), "
                        "NOW(), :status, 0, 8, NOW())"
                    ),
                    {
                        "operation": OutboxOperation.SET_DOCUMENT_ID.value,
                        "status": OutboxStatus.PENDING.value,
                        "doc_id": doc_id,
                        "source": committed_source or source,
                    },
                )


def apply_revision(connection, direction):
    revision = importlib.import_module(REVISION_MODULE)
    with Operations.context(MigrationContext.configure(connection)):
        getattr(revision, direction)()


@pytest.mark.parametrize("key", [FIRST_KEY, "docs/раздел/manual #?.md", "docs/a//manual.md"])
async def test_migration_backfills_exact_key_and_document_binding(pg_engine, key):
    await prepare_legacy_schema(pg_engine)
    await add_legacy_rows(pg_engine, source=f"s3://bucket/{key}")
    async with pg_engine.begin() as connection:
        await connection.run_sync(apply_revision, "upgrade")
        registry_key = await connection.scalar(text("SELECT object_key FROM ingestion_registry"))
        document_key = await connection.scalar(text("SELECT ingestion_key FROM documents"))
        assert registry_key == key
        assert document_key == key
        assert await connection.scalar(text("SELECT file_hash FROM ingestion_registry")) == "hash"
        await connection.run_sync(apply_revision, "downgrade")
        assert await connection.scalar(text("SELECT filename FROM ingestion_registry")) == FILENAME


async def test_migration_invalidates_registry_without_document(pg_engine):
    await prepare_legacy_schema(pg_engine)
    await add_legacy_rows(pg_engine, source=f"s3://bucket/{FIRST_KEY}", missing_document=True)
    async with pg_engine.begin() as connection:
        await connection.run_sync(apply_revision, "upgrade")
        assert await connection.scalar(text("SELECT file_hash FROM ingestion_registry")) == ""
        assert await connection.scalar(text("SELECT object_key FROM ingestion_registry")) == FIRST_KEY


async def test_migration_rejects_ambiguous_legacy_document_binding(pg_engine):
    await prepare_legacy_schema(pg_engine)
    await add_legacy_rows(pg_engine, source=f"s3://bucket/{FIRST_KEY}", ambiguous=True)
    with pytest.raises(RuntimeError, match="ambiguous"):
        async with pg_engine.begin() as connection:
            await connection.run_sync(apply_revision, "upgrade")
    async with pg_engine.connect() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM documents")) == 2


async def test_migration_rejects_invalid_legacy_source(pg_engine):
    await prepare_legacy_schema(pg_engine)
    await add_legacy_rows(pg_engine, source="manual.md")
    with pytest.raises(RuntimeError, match="S3 source"):
        async with pg_engine.begin() as connection:
            await connection.run_sync(apply_revision, "upgrade")


async def test_migration_recovers_registry_lag_from_committed_source(pg_engine):
    await prepare_legacy_schema(pg_engine)
    await add_legacy_rows(
        pg_engine, source=f"s3://bucket/{FIRST_KEY}", committed_source=f"s3://bucket/{SECOND_KEY}"
    )
    async with pg_engine.begin() as connection:
        await connection.run_sync(apply_revision, "upgrade")
        assert await connection.scalar(text("SELECT ingestion_key FROM documents")) == SECOND_KEY
        assert await connection.scalar(text("SELECT object_key FROM ingestion_registry")) == FIRST_KEY
        assert await connection.scalar(text("SELECT file_hash FROM ingestion_registry")) == ""


async def test_migration_rejects_document_without_source_evidence(pg_engine):
    await prepare_legacy_schema(pg_engine)
    await add_legacy_rows(pg_engine, source=f"s3://bucket/{FIRST_KEY}", source_evidence=False)
    with pytest.raises(RuntimeError, match="source evidence"):
        async with pg_engine.begin() as connection:
            await connection.run_sync(apply_revision, "upgrade")


async def test_downgrade_rejects_duplicate_basenames(pg_engine):
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        repository = SQLAlchemyIngestionRegistryRepository(session)
        for key in (FIRST_KEY, SECOND_KEY):
            await repository.upsert(
                IngestionRegistryEntry(key, FILENAME, "hash", f"s3://bucket/{key}", 1, 10)
            )
    with pytest.raises(RuntimeError, match="duplicate basenames"):
        async with pg_engine.begin() as connection:
            await connection.run_sync(apply_revision, "downgrade")


async def test_downgrade_rejects_source_exceeding_legacy_limit(pg_engine):
    key = "docs/" + "a" * 1009 + "/manual.md"
    sessions = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with sessions.begin() as session:
        await SQLAlchemyIngestionRegistryRepository(session).upsert(
            IngestionRegistryEntry(key, FILENAME, "hash", f"s3://bucket/{key}", 1, 10)
        )
    with pytest.raises(RuntimeError, match="long ingestion source"):
        async with pg_engine.begin() as connection:
            await connection.run_sync(apply_revision, "downgrade")
