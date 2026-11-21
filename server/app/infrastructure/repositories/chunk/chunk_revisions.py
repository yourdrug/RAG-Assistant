"""Staged chunk revisions and their publication, using the caller's session."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING
import uuid

from sqlalchemy import delete, insert, literal, select

from domain.knowledge.chunk import ChunkRecord, extract_chunk_context
from domain.knowledge.document_nodes import DocumentNode
from domain.knowledge.enums import DocumentVisibility, DomainKey
from domain.knowledge.repositories.chunk import DEFAULT_CHUNK_INSERT_OPTIONS
from domain.knowledge.revision import RevisionSwitch, SourceVersion, StagedRevision
from infrastructure.database.models import ChunkModel, DocumentStagedRevisionModel, StagedChunkModel
from infrastructure.repositories.chunk.document_nodes import SQLAlchemyDocumentNodeRepository


if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy.ext.asyncio import AsyncSession

    from domain.knowledge.chunk import ChunkInsert
    from domain.knowledge.repositories.chunk import ChunkInsertOptions


# Columns copied verbatim from a staged chunk to the published chunk with the same id.
PUBLISHED_COLUMNS = (
    "id",
    "document_id",
    "chunk_index",
    "content",
    "embed_text",
    "filename",
    "visibility",
    "doc_domain",
    "owner_id",
    "group_id",
    "space_id",
    "content_hash",
    "domain_metadata",
    "context_metadata",
    "pack_metadata",
    "section",
    "heading",
    "heading_level",
    "content_type",
    "doc_title",
    "doc_type",
)


def staged_record(orm: StagedChunkModel) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=orm.id,
        document_id=orm.document_id,
        filename=orm.filename,
        content=orm.content,
        embed_text=orm.embed_text,
        chunk_index=orm.chunk_index,
        visibility=DocumentVisibility.parse(orm.visibility),
        doc_domain=DomainKey(orm.doc_domain),
        owner_id=orm.owner_id,
        group_id=orm.group_id,
        creation_date=orm.creation_date,
        content_hash=orm.content_hash,
        section=orm.section,
        heading=orm.heading,
        heading_level=orm.heading_level,
        content_type=orm.content_type,
        doc_title=orm.doc_title,
        doc_type=orm.doc_type,
        pack_metadata=orm.pack_metadata or {},
        context_metadata=orm.context_metadata or {},
    )


def revision_source_version(revision: DocumentStagedRevisionModel) -> SourceVersion | None:
    if revision.source_version is None:
        return None

    return SourceVersion(revision.source_version, revision.source_revision)


def node_payload(node: DocumentNode) -> dict[str, Any]:
    payload = asdict(node)
    payload["pages"] = list(node.pages)
    return payload


def node_from_payload(payload: dict[str, Any]) -> DocumentNode:
    return DocumentNode(**{**payload, "pages": tuple(payload.get("pages") or ())})


class SQLAlchemyChunkRevisionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._nodes = SQLAlchemyDocumentNodeRepository(session)

    async def stage_revision(
        self,
        document_id: int,
        filename: str,
        visibility: str,
        chunks: list[ChunkInsert],
        *,
        nodes: list[DocumentNode],
        options: ChunkInsertOptions = DEFAULT_CHUNK_INSERT_OPTIONS,
        source_version: SourceVersion | None = None,
    ) -> StagedRevision:
        token = uuid.uuid4().hex
        self._session.add(
            DocumentStagedRevisionModel(
                document_id=document_id,
                token=token,
                source_version=source_version.version if source_version else None,
                source_revision=source_version.revision if source_version else None,
                nodes=[node_payload(node) for node in nodes],
            )
        )
        await self._session.flush()
        models = [
            StagedChunkModel(
                document_id=document_id,
                chunk_index=index,
                content=chunk.content,
                embed_text=chunk.embed_text,
                filename=filename,
                visibility=visibility,
                doc_domain=options.doc_domain,
                owner_id=options.owner_id,
                group_id=options.group_id,
                space_id=options.space_id,
                content_hash=chunk.content_hash,
                domain_metadata=options.domain_metadata,
                pack_metadata=dict(options.pack_metadata or {}),
                section=chunk.section,
                heading=chunk.heading,
                heading_level=chunk.heading_level,
                content_type=chunk.content_type,
                doc_title=chunk.doc_title,
                doc_type=chunk.doc_type,
                context_metadata=extract_chunk_context(chunk.context_metadata),
            )
            for index, chunk in enumerate(chunks)
        ]
        self._session.add_all(models)
        await self._session.flush()
        return StagedRevision(document_id, token, tuple(model.id for model in models), source_version)

    async def get_staged_revision(self, document_id: int) -> StagedRevision | None:
        revision = (
            await self._session.execute(
                select(DocumentStagedRevisionModel).where(
                    DocumentStagedRevisionModel.document_id == document_id
                )
            )
        ).scalar_one_or_none()

        if revision is None:
            return None

        ids = await self._session.execute(
            select(StagedChunkModel.id)
            .where(StagedChunkModel.document_id == document_id)
            .order_by(StagedChunkModel.chunk_index)
        )
        return StagedRevision(
            document_id, revision.token, tuple(ids.scalars().all()), revision_source_version(revision)
        )

    async def staged_source_versions(self, document_ids: list[int]) -> dict[int, SourceVersion]:
        if not document_ids:
            return {}

        result = await self._session.execute(
            select(DocumentStagedRevisionModel).where(
                DocumentStagedRevisionModel.document_id.in_(document_ids),
                DocumentStagedRevisionModel.source_version.is_not(None),
            )
        )
        return {
            revision.document_id: version
            for revision in result.scalars().all()
            if (version := revision_source_version(revision)) is not None
        }

    async def get_staged_by_ids(
        self, document_id: int, token: str, chunk_ids: list[int]
    ) -> list[ChunkRecord]:
        if not chunk_ids:
            return []

        result = await self._session.execute(
            select(StagedChunkModel)
            .join(
                DocumentStagedRevisionModel,
                DocumentStagedRevisionModel.document_id == StagedChunkModel.document_id,
            )
            .where(
                StagedChunkModel.document_id == document_id,
                DocumentStagedRevisionModel.token == token,
                StagedChunkModel.id.in_(chunk_ids),
            )
        )
        return [staged_record(orm) for orm in result.scalars().all()]

    async def discard_staged_revision(self, document_id: int) -> list[int]:
        ids = await self._session.execute(
            select(StagedChunkModel.id).where(StagedChunkModel.document_id == document_id)
        )
        staged_ids = list(ids.scalars().all())
        # Staged chunks cascade with their revision row.
        await self._session.execute(
            delete(DocumentStagedRevisionModel).where(DocumentStagedRevisionModel.document_id == document_id)
        )
        await self._session.flush()
        return staged_ids

    async def publish_staged_revision(self, document_id: int, token: str) -> RevisionSwitch | None:
        revision = (
            await self._session.execute(
                select(DocumentStagedRevisionModel)
                .where(DocumentStagedRevisionModel.document_id == document_id)
                .with_for_update()
            )
        ).scalar_one_or_none()

        if revision is None or revision.token != token:
            return None

        doc_domain = (
            await self._session.execute(
                select(StagedChunkModel.doc_domain)
                .where(StagedChunkModel.document_id == document_id)
                .limit(1)
            )
        ).scalar_one_or_none()
        replaced = await self._session.execute(
            select(ChunkModel.id).where(ChunkModel.document_id == document_id)
        )
        replaced_ids = tuple(replaced.scalars().all())
        await self._session.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
        staged_columns = [getattr(StagedChunkModel, name) for name in PUBLISHED_COLUMNS]
        published = await self._session.execute(
            insert(ChunkModel)
            .from_select(
                [*PUBLISHED_COLUMNS, "manual"],
                select(*staged_columns, literal(value=False)).where(
                    StagedChunkModel.document_id == document_id
                ),
            )
            .returning(ChunkModel.id)
        )
        published_ids = tuple(sorted(published.scalars().all()))
        await self._nodes.replace_document_nodes(
            document_id, [node_from_payload(payload) for payload in revision.nodes or []]
        )
        source_version = revision_source_version(revision)
        await self._session.delete(revision)
        await self._session.flush()
        return RevisionSwitch(
            published_ids=published_ids,
            replaced_ids=replaced_ids,
            source_version=source_version,
            doc_domain=doc_domain,
        )
